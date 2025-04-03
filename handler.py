import os
import time
import torch
import base64
import json
import numpy as np
import runpod
from io import BytesIO
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
            "audio": "<base64_encoded_audio>",
            "sample_rate": 16000,  # Optional, defaults to 16000
            "return_embeddings": true  # Optional, defaults to true
        }
    }
    """
    job_input = event["input"]
    
    # Extract input parameters
    waveform = job_input.get("waveform")
    sr = job_input.get("sr")
    
    if not waveform or not sr:
        return {"error": "No audio data provided"}
    
    try:
        # Process audio data
        start_time = time.time()
        
        # Decode audio
        # audio_tensor, sr = decode_audio(encoded_audio, sample_rate)
        
        # Prepare input for pipeline
        input_dict = {
            'waveform': waveform,  # Add batch dimension
            'sample_rate': sr
        }
        
        # Run diarization
        diarization, embeddings = pipeline(input_dict, 
                                       return_embeddings=True)
        
        # Prepare the response
        response = {
            "diarization": format_diarization_output(diarization),
            "embeddings": embeddings.cpu().numpy().tolist() if embeddings is not None else None
        }
        
        processing_time = time.time() - start_time
        response["processing_time"] = processing_time
        
        return response
    
    except Exception as e:
        return {"error": str(e)}


# Initialize the model
init()

# Start the RunPod handler
runpod.serverless.start({"handler": handler})
