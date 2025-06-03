import requests
import tempfile
import logging
import os
import torch
import soundfile as sf
import base64
import io

logger = logging.getLogger(__name__)

# If the same speaker is talking for less than JOIN_SEGMENT_THRESHOLD seconds, join the segments
JOIN_SEGMENT_THRESHOLD = float(os.getenv("JOIN_SEGMENT_THRESHOLD", 2))

# Number of seconds of speech to consider a cross-talk
CROSS_TALK_DURATION_THRESHOLD = float(os.getenv("CROSS_TALK_DURATION_THRESHOLD", 2))

# Number of seconds of gap between current speech and adjecent speech to consider a cross-talk
CROSS_TALK_GAP_THRESHOLD = float(os.getenv("CROSS_TALK_GAP_THRESHOLD", 5))

def download_url_to_mp3(url):
    """
    Downloads audio content from a given URL and saves it as an MP3 file.

    Args:
        url (str): The URL of the audio file to download.

    Returns:
        str: Path to the temporary MP3 file where the audio is saved.

    Raises:
        Exception: If the download fails or the URL is invalid.
    """
    logger.debug(f"Downloading audio from URL: {url}")
    try:
        response = requests.get(url)
        logger.debug(f"Download response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Failed to download file from URL: {url}")
            raise Exception("Failed to download file from URL")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.write(response.content)
        temp_file.close()
        logger.debug(f"Audio downloaded and saved to tempfile: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logger.error(f"Error in download_url_to_mp3: {e}")
        raise

def format_diarization_output(diarization):
    """
    Formats the output of a diarization model into a list of speaker segments.

    Args:
        diarization (Annotation): A pyannote.core.Annotation object containing speaker diarization results.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - speaker (str): Speaker label
            - start (float): Start time of the segment in seconds
            - end (float): End time of the segment in seconds
    """
    logger.debug("Starting to format diarization output")
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        })
    logger.debug(f"Formatted {len(segments)} segments from diarization output")
    return segments

def join_diarization_output(segments):
    """
    Joins consecutive segments from the same speaker that are within a threshold time gap.

    Args:
        segments (list): List of dictionaries containing speaker segments, where each dictionary has:
            - speaker (str): Speaker label
            - start (float): Start time of the segment
            - end (float): End time of the segment

    Returns:
        list: List of merged segments with the same structure as input, plus:
            - duration (float): Duration of each segment in seconds
    """
    if not segments:
        logger.debug("No segments to join")
        return segments
    
    logger.debug(f"Starting to join {len(segments)} segments with threshold {JOIN_SEGMENT_THRESHOLD}s")
    joined_segments = []
    current_segment = segments[0].copy()
    
    for i in range(1, len(segments)):
        next_segment = segments[i]
        
        if (current_segment["speaker"] == next_segment["speaker"] and 
            next_segment["start"] - current_segment["end"] < JOIN_SEGMENT_THRESHOLD):
            current_segment["end"] = next_segment["end"]
            logger.debug(f"Joined segments for speaker {current_segment['speaker']}")
        else:
            current_segment["duration"] = current_segment["end"] - current_segment["start"]
            joined_segments.append(current_segment)
            current_segment = next_segment.copy()
    
    current_segment["duration"] = current_segment["end"] - current_segment["start"]
    joined_segments.append(current_segment)
    
    logger.debug(f"Joined into {len(joined_segments)} segments")
    return joined_segments

def detect_cross_talk(segments):
    """
    Detects potential cross-talk segments based on duration and gap thresholds.

    Args:
        segments (list): List of dictionaries containing speaker segments, where each dictionary has:
            - speaker (str): Speaker label
            - start (float): Start time
            - end (float): End time
            - duration (float): Duration of the segment

    Returns:
        int: Number of detected cross-talk segments
    """
    logger.debug(f"Starting cross-talk detection with {len(segments)} segments")
    cross_talk_segments = []
    
    for i in range(len(segments)):
        current_segment = segments[i]
        
        if current_segment["duration"] < CROSS_TALK_DURATION_THRESHOLD:
            prev_segment = segments[i-1] if i > 0 else None
            next_segment = segments[i+1] if i < len(segments)-1 else None
            
            prev_gap = (current_segment["start"] - prev_segment["end"]) if prev_segment else float('inf')
            next_gap = (next_segment["start"] - current_segment["end"]) if next_segment else float('inf')
            
            if prev_gap < CROSS_TALK_GAP_THRESHOLD or next_gap < CROSS_TALK_GAP_THRESHOLD:
                cross_talk = current_segment.copy()
                cross_talk["prev_speaker"] = prev_segment["speaker"] if prev_segment else None
                cross_talk["next_speaker"] = next_segment["speaker"] if next_segment else None
                cross_talk["prev_gap"] = prev_gap if prev_gap != float('inf') else None
                cross_talk["next_gap"] = next_gap if next_gap != float('inf') else None
                cross_talk_segments.append(cross_talk)
                logger.debug(f"Detected cross-talk at {current_segment['start']:.2f}s")
    
    logger.info(f"Detected {len(cross_talk_segments)} cross-talk segments")
    return len(cross_talk_segments)

def calculate_amplitude_ratio(waveform, sample_rate, segments):
    """
    Calculates the ratio of average amplitude between non-speech and speech segments.

    Args:
        waveform (torch.Tensor): Audio waveform tensor
        sample_rate (int): Sampling rate of the audio in Hz
        segments (list): List of dictionaries containing speaker segments, where each dictionary has:
            - start (float): Start time in seconds
            - end (float): End time in seconds

    Returns:
        float: Ratio of non-speech to speech amplitude (non_speech_amplitude / speech_amplitude)
    """
    logger.debug(f"Calculating amplitude ratio for audio with sample rate {sample_rate}Hz")
    
    if len(waveform.shape) == 2:
        waveform = waveform.squeeze(0)
    
    amplitude = torch.abs(waveform)
    speech_samples = []
    non_speech_samples = []
    current_sample = 0
    
    for segment in segments:
        start_sample = int(segment["start"] * sample_rate)
        end_sample = int(segment["end"] * sample_rate)
        
        if start_sample > current_sample:
            non_speech_samples.append(amplitude[current_sample:start_sample])
        
        if end_sample <= len(amplitude):
            speech_samples.append(amplitude[start_sample:end_sample])
        
        current_sample = end_sample
    
    if current_sample < len(amplitude):
        non_speech_samples.append(amplitude[current_sample:])
    
    speech_amplitude = torch.mean(torch.cat(speech_samples)).item() if len(speech_samples)>0 else 0
    non_speech_amplitude = torch.mean(torch.cat(non_speech_samples)).item() if len(non_speech_samples)>0 and speech_amplitude!=0 else 0
    
    ratio = non_speech_amplitude / (speech_amplitude + 1e-10)
    logger.debug(f"Calculated amplitude ratio: {ratio:.4f}")
    
    return ratio

def get_pyannote_input_dict(audio_url, audio_bytes):
    if audio_url:
        audio_path = download_url_to_mp3(audio_url)
        waveform, sample_rate = sf.read(audio_path)
        os.remove(audio_path)
    elif audio_bytes:
        audio_bytes = base64.b64decode(audio_bytes)
        audio_io = io.BytesIO(audio_bytes)
        waveform, sample_rate = sf.read(audio_io)
    else:
        raise ValueError("Either audio_path or audio_bytes must be provided")
    waveform = torch.from_numpy(waveform).float()
    # Handle mono vs stereo
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    elif len(waveform.shape) == 2:
        if waveform.shape[1] < waveform.shape[0]:
            waveform = waveform.t()
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    input_dict = {
        'waveform': waveform,
        'sample_rate': sample_rate
    }
    return input_dict

def calculate_silence_ratio(waveform, sample_rate, joined_diarization):
    total_audio_duration =  waveform.shape[1]/ sample_rate
    total_speech_duration = sum([x['duration'] for x in joined_diarization])
    return 1 - (total_speech_duration / total_audio_duration)