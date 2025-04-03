# PyAnnote Speaker Diarization RunPod Handler

This repository contains a RunPod handler for speaker diarization using PyAnnote. It takes audio input and returns diarization results with speaker segments and optionally speaker embeddings.

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
    "audio": "<base64_encoded_audio>",
    "sample_rate": 16000,
    "return_embeddings": true
  }
}
```

### Parameters

- `audio`: Required. Base64-encoded audio data in float32 format
- `sample_rate`: Optional. Sample rate of the audio (default: 16000)
- `return_embeddings`: Optional. Whether to return speaker embeddings (default: true)

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
  "embeddings": [...],  // Only if return_embeddings is true
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