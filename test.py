import requests
from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv
import torch
import soundfile as sf
import base64
load_dotenv()

## Local Build##

# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.1",
#     use_auth_token=os.getenv("HF_TOKEN"),
#     )



# # send pipeline to GPU (when available)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pipeline.to(device)

# #read waveform
# # sample first 10 seconds
# waveform, sample_rate = sf.read("audio.wav")
# waveform = waveform[:sample_rate*20]
# waveform = torch.from_numpy(waveform).float()

# input_dict = {
#         'waveform': waveform[None],
#         'sample_rate': sample_rate
#     }

# diarization, embeddings = pipeline(input_dict, return_embeddings=True)

# labels = diarization.labels()

# for label,embedding in zip(labels,embeddings):
#     print({label:embedding.tolist()})

# print('***')
# #print the result
# for turn, _, speaker in diarization.itertracks(yield_label=True):
#     print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

## /Local Build##



# ## Local API Call ##
# import pandas as pd
# with open("audio.wav", "rb") as audio_file:
#     audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
#     # take first 10 seconds
#     audio_base64 = audio_base64[:10*16000*2]
# json_input = {
#     "input": {
#         "audio_data": audio_base64,
#         "file_type": "wav"
#     }
# }

# response = requests.post("http://localhost:8080/runsync", 
#                          json=json_input)
# print(response.json())

# dirz_df = pd.DataFrame(response.json()['output']['diarization'])
# ## /Local API Call ##




with open("audio.wav", "rb") as audio_file:
    audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {os.getenv("RUNPOD_API_KEY")}'
}

json_input = {
    "input": {
        "audio_data": audio_base64,
        "file_type": "wav"
    }
}

response = requests.post('https://api.runpod.ai/v2/rqgkup34l95eph/runsync', 
                         headers=headers, 
                         json=json_input)

print(response.json())