# PyAnnote Speaker Diarization RunPod Handler

This repository contains a RunPod handler for speaker diarization using PyAnnote. It takes audio input and returns diarization results with speaker segments and speaker embeddings.

## Setup

### Prerequisites

1. A Hugging Face account with access to the PyAnnote model
2. RunPod account with API access
3. Docker installed locally (for building the image)

### Environment Variables

Set the following environment variable when deploying:

- `HF_TOKEN`: Your Hugging Face API token with access to the PyAnnote model

## Building the Docker Image

```bash
docker build -t your-username/diarization-handler:latest .
docker push your-username/diarization-handler:latest
```

## Deploying on RunPod

1. Go to your RunPod Serverless dashboard
2. Create a new endpoint using your Docker image
3. Set the required environment variables
4. Deploy your endpoint

## API Usage

Send a POST request to your RunPod endpoint with the following structure:

```json
{
  "input": {
    "audio_data": "<base64_encoded_audio>",
    "file_type": "wav"
  }
}
```

### Parameters

- `audio_data`: Required. Base64-encoded audio file (wav, mp3, etc.)
- `file_type`: Optional. File format of the audio (default: "wav")

### Response

```json
{
  "diarization": [
    {
      "speaker": "SPEAKER_0",
      "start": 0.0,
      "end": 2.5
    },
    {
      "speaker": "SPEAKER_1",
      "start": 2.7,
      "end": 5.2
    }
  ],
  "embeddings_dict": {
    "SPEAKER_0": [0.1, 0.2, ...],
    "SPEAKER_1": [0.3, 0.4, ...]
  },
  "processing_time": 3.45
}
```

## Local Testing

To test locally before deploying:

```bash
# Export your HF token
export HF_TOKEN="your_huggingface_token"

# Run the handler locally
python handler.py
```

You can then test it with:

```bash
python test.py
```

## Example Code

```python
import requests
import base64
import os

# Read and encode audio file
with open("audio.wav", "rb") as audio_file:
    audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

# Set up headers with RunPod API key
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {os.getenv("RUNPOD_API_KEY")}'
}

# Prepare the request payload
json_input = {
    "input": {
        "audio_data": audio_base64,
        "file_type": "wav"
    }
}

# Send the request to your RunPod endpoint
response = requests.post('https://api.runpod.ai/v2/your-endpoint-id/runsync', 
                         headers=headers, 
                         json=json_input)

# Process the response
result = response.json()
```