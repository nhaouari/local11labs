# Local11Labs: High-Quality Text-to-Speech & Podcast Generator

Local11Labs is a powerful text-to-speech and podcast generation tool powered by the lightweight [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model. Generate natural-sounding speech and multi-speaker podcasts locally on your machine.

### Key Changes

1. **Gradio WebUI Integration**
   - Integrated Gradio WebUI into the application to enhance user interaction and accessibility.

2. **Text-to-Speech (TTS) Tab**
   - **Input Options**: Users can provide input via a text field or upload a `.txt` file.
   - **Speech Rhythm Control**: Adjust the rhythm and pacing of generated speech.
   - **Voices Dropdown Menu**: Select from a variety of available voice profiles.
   - **Device Selection**: Automatic device detection with the option to specify CUDA or CPU.

3. **Podcast Dialogue Generation Tab (Powered by Gemini)**
   - Enables dynamic generation of podcast dialogues.
   - Input customizable host names to create unique and personalized conversations.

4. **Podcast Audio Generation**
   - **Script Input Options**: 
      - Use the JSON output from the Dialogue Generation tab.
      - Upload a custom JSON file.
      - Directly edit or input text within the interface.
   - **Dynamic Host Mapping**:
      - Assign available voices to host names.
      - Add as many hosts as needed without any limitations.
   - **Speech Rhythm Control**: Fine-tune the speech rhythm for each host to enhance the audio experience.


## Key Features
- ⚡ Fast and efficient text-to-speech generation
- 🎙️ Multi-speaker podcast creation with distinct voices
- 📝 Smart text chunking for handling long content
- 🎛️ Customizable voice profiles with caching
- 🔊 Professional audio quality with natural pauses and transitions
- 🚀 Easy to use with minimal setup required





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

## Webui demo :
Notebook will be added soon
for not you can clone it and run it.


## TODO List
- [ ] Add REST API endpoints for remote TTS generation
- [ ] Implement streaming audio support
- [x] Create web interface for easy usage
- [ ] Add Docker support for easy deployment


## Screenshots :
![screen_shot1](/ui-sceenshot1.png)
