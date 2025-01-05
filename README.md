# Local11Labs: High-Quality Text-to-Speech & Podcast Generator

Local11Labs is a powerful text-to-speech and podcast generation tool powered by the lightweight [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model. Generate natural-sounding speech and multi-speaker podcasts locally on your machine.

## Key Features
- ⚡ Fast and efficient text-to-speech generation
- 🎙️ Multi-speaker podcast creation with distinct voices
- 📝 Smart text chunking for handling long content
- 🎛️ Customizable voice profiles with caching
- 🔊 Professional audio quality with natural pauses and transitions
- 🚀 Easy to use with minimal setup required

## Performance (Colab T4 GPU FP16)

Local11Labs deliver exceptional performance, ensuring fast and efficient text-to-speech (TTS) and podcast generation without compromising quality. Below are the key performance metrics measured using a Colab T4 GPU with FP16 precision:

| **Metric**               | **Value**                     |
|--------------------------|-------------------------------|
| **Total processing time**| 27.17 seconds                 |
| **Processing speed**     | 788.0 characters/second       |
| **Text length**          | 21,414 characters             |
| **Audio length**         | 1,271.9 seconds (21.2 minutes)|
| **Average speaking rate**| 16.8 characters/second        |



## Podcast demo 

https://github.com/user-attachments/assets/fdb09bfb-48d6-4083-852d-c4353b4df2a9

## Quick Start
https://github.com/user-attachments/assets/3bc23c3f-86f0-4aad-a34a-6a16cc3aa80a


### Text-to-Speech Demo
Try out basic text-to-speech generation in our interactive Colab notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mi3IewrWoHunNEmPXcItCLom6Je8TeUw?usp=sharing)

### Podcast Generation Demo
Create multi-speaker podcasts using our podcast generation notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1v8pGGGLGEPhIva0Jq-YcrL1XYuZm_hYA?usp=sharing)

## TODO List
- [ ] Add REST API endpoints for remote TTS generation
- [ ] Implement streaming audio support
- [ ] Create web interface for easy usage
- [ ] Add Docker support for easy deployment
