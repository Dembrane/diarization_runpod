import os
import time
import torch
import runpod
import base64
import io
import soundfile as sf
import numpy as np
from pyannote.audio import Pipeline


# Global variables
pipeline = None
device = None


def init():
    """Initialize the diarization pipeline."""
    global pipeline, device
    
    # Get HF token from environment variables
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    # Load the diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    
    # Set the device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    
    print(f"Diarization pipeline loaded successfully on {device}")


def format_diarization_output(diarization):
    """Format diarization results to a serializable format."""
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        })
    return segments


def handler(event):
    """
    RunPod handler function for processing diarization requests.
    
    Expected input structure:
    {
        "input": {
            "audio_data": "<base64_encoded_audio>",
            "file_type": "<file_extension>"
        }
    }
    """
    job_input = event["input"]
    audio_data = job_input.get("audio_data")
    file_type = job_input.get("file_type", "wav")  # default to wav if not specified
    
    if audio_data is None:
        return {"error": "No audio data provided"}
    
    try:
        # Decode base64 to audio
        audio_bytes = base64.b64decode(audio_data)
        audio_io = io.BytesIO(audio_bytes)
        
        # Read audio file using soundfile
        waveform, sample_rate = sf.read(audio_io)
        
        # Convert to float tensor
        waveform = torch.from_numpy(waveform).float()
        
        # Add batch dimension if needed (pyannote expects [channel, time])
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        elif len(waveform.shape) == 2:
            # If stereo, convert to mono by averaging channels
            waveform = waveform.mean(axis=0, keepdim=True)
        
        # Process audio data
        start_time = time.time()

        # Prepare input for pipeline
        input_dict = {
            'waveform': waveform,
            'sample_rate': sample_rate
        }
        
        # Run diarization
        diarization, embeddings = pipeline(input_dict, return_embeddings=True)
        
        # Prepare the response
        response = {
            "diarization": format_diarization_output(diarization),
            "embeddings": embeddings.tolist() if embeddings is not None else None
        }
        
        processing_time = time.time() - start_time
        response["processing_time"] = processing_time
        
        return response
    
    except Exception as e:
        return {"error": f"Error processing audio: {str(e)}"}


# Initialize the model
init()

# Start the RunPod handler
runpod.serverless.start({"handler": handler})
