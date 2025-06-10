from diarizers import SegmentationModel
import requests
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.pipeline import Optimizer
import os
from dotenv import load_dotenv
import torch
import soundfile as sf
import base64
import tempfile
import logging
import time
import pickle
from utils import *
import io

load_dotenv()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "HF_TOKEN is not set"

logger = logging.getLogger(__name__)
# audio_url = "https://github.com/runpod-workers/sample-inputs/raw/refs/heads/main/audio/Arthur.mp3"
audio_url ='https://github.com/Dembrane/diarization_runpod/raw/refs/heads/main/test_dutch.mp3'

input_dict = get_pyannote_input_dict(audio_url, None)
waveform, sample_rate = input_dict["waveform"], input_dict["sample_rate"]

# Load your custom segmentation model

# Create custom pipeline with your segmentation model
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline.to(device)
model = SegmentationModel().from_pretrained("Roy2358/speaker-segmentation-fine-tuned-nl")
model = model.to_pyannote_model()

# pipeline._segmentation.model = model.to(device)

# Run diarization
diarization, embeddings = pipeline(input_dict, return_embeddings=True)

labels = diarization.labels()

formatted_diarization = format_diarization_output(diarization)
joined_diarization = join_diarization_output(formatted_diarization)
noise_ratio = calculate_amplitude_ratio(waveform, sample_rate, joined_diarization)
cross_talk_instances = detect_cross_talk(joined_diarization)
silence_ratio = calculate_silence_ratio(waveform, sample_rate, joined_diarization)
print(joined_diarization)


