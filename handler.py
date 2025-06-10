import os
import time
import torch
import runpod
import base64
import io
import soundfile as sf
import numpy as np
from pyannote.audio import Pipeline
from runpod.serverless.utils import download_files_from_urls
from utils import get_pyannote_input_dict, format_diarization_output, join_diarization_output, calculate_amplitude_ratio, detect_cross_talk, calculate_silence_ratio
import logging
from diarizers import SegmentationModel
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.pipeline import Optimizer

logger = logging.getLogger(__name__)


hf_token = os.environ.get("HF_TOKEN")
assert hf_token, "HF_TOKEN is not set"

def init():
    """Initialize the diarization pipeline."""
    global pipeline, device, segmentation_model_dict    
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    logger.info(f"Diarization pipeline loaded successfully on {device}")

    segmentation_model_dict={'nl': SegmentationModel().from_pretrained("Roy2358/speaker-segmentation-fine-tuned-nl").to_pyannote_model()}


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
    job_id = event["id"]
    audio_data = job_input.get("audio_data")
    audio_url = job_input.get("audio")
    language = job_input.get("language")
    
    try:
        if language in segmentation_model_dict:
            logger.debug(f"Using segmentation model for {language}")
            pipeline._segmentation.model = segmentation_model_dict[language].to(device)
        else:
            logger.debug(f"Using default segmentation model")
        input_dict = get_pyannote_input_dict(audio_url, audio_data)
        waveform, sample_rate = input_dict["waveform"], input_dict["sample_rate"]
        diarization, embeddings = pipeline(input_dict, return_embeddings=True) 
        formatted_diarization = format_diarization_output(diarization)
        joined_diarization = join_diarization_output(formatted_diarization)
        noise_ratio = calculate_amplitude_ratio(waveform, sample_rate, joined_diarization)
        cross_talk_instances = detect_cross_talk(joined_diarization)
        silence_ratio = calculate_silence_ratio(waveform, sample_rate, joined_diarization)
        
        return {
            "noise_ratio": noise_ratio,
            "cross_talk_instances": cross_talk_instances,
            "silence_ratio": silence_ratio,
            "joined_diarization": joined_diarization,
            # "embeddings": embeddings,
        }
    except Exception as e:
        return {"error": f"Error processing audio: {str(e)}"}


# Initialize the model
init()

# Start the RunPod handler
runpod.serverless.start({"handler": handler})
