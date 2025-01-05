import sys
import os
import torch
import gradio as gr
import tempfile


kokoro_path = os.path.abspath('Kokoro-82M')
print(f"kokoro_path :{kokoro_path}")
if kokoro_path not in sys.path:
    sys.path.insert(0, kokoro_path)

try:
    from kokoro import generate
    from models import build_model
except ImportError:
    print("Kokoro not found. Please install the Kokoro-82M package.")
    generate = None


# Add the 'src' directory to the Python path
src_path = os.path.abspath('src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.long_speech_generation import generate_long_text_optimized
from src.podcast_generation import generate_audio
from src.utils import read_file_content, process_long_text, generate_audio_enhanced

# Import the Tab UI components
from podcast_tab import create_podcast_tab
from text_to_speech_tab import create_text_to_speech_tab




VOICEPACK_DIR = os.path.join(kokoro_path, "voices")

MODELS_LIST = {
    "v0_19-full-fp32": os.path.join(kokoro_path, "kokoro-v0_19.pth"),
    "v0_19-half-fp16": os.path.join(kokoro_path, "fp16/kokoro-v0_19-half.pth"),
}

# Available voices
CHOICES = {
    '🇺🇸 🚺 American Female ⭐': 'af',
    '🇺🇸 🚺 Bella ⭐': 'af_bella',
    '🇺🇸 🚺 Sarah ⭐': 'af_sarah',
    '🇺🇸 🚹 Michael ⭐': 'am_michael',
    '🇺🇸 🚹 nicole': 'af_nicole',
    '🇺🇸 🚹 sky': 'af_sky',
    '🇺🇸 🚹 Adam': 'am_adam',
    '🇬🇧 🚺 British Female emma': 'bf_emma',
    '🇬🇧 🚺 British Female isabella': 'bf_isabella',
    '🇬🇧 🚹 British Male george': 'bm_george',
    '🇬🇧 🚹 British Male lewis': 'bm_lewis',

}

# Device Selection
device_options = ["auto", "cpu", "cuda"]

# Initialize model and voices (lazy loading)
MODEL_NAME = None
MODEL = None
MODEL_DEVICE = None
VOICES = {}

# Text normalization functions (simplified)
def normalize_text(text):
    text = text.replace("’", "'")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

SAMPLE_RATE = 24000

def load_model_and_voice(selected_device, model_path, voice):
    global MODEL, VOICES, MODELS_LIST, MODEL_NAME, MODEL_DEVICE
    try:
        if selected_device == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif selected_device == "cuda":
            if torch.cuda.is_available():
                print("CUDA is available. Using GPU.")
                device = 'cuda'
            else:
                print("CUDA is not available. Using CPU instead.")
                device = 'cpu'
        else:
            device = 'cpu'
    except Exception as e:
        print("CUDA Error is not available. Using CPU instead.")
        device = 'cpu'

    # Check if we need to reload the model
    should_reload = (
        MODEL is None or
        MODEL_DEVICE != device or
        MODEL_NAME != model_path
    )

    if should_reload:
        MODEL = build_model(model_path, device)
        MODEL_NAME = model_path
        MODEL_DEVICE = device
        print(f"Loaded model {model_path} on {device}")

    if voice not in VOICES:
        VOICES[voice] = torch.load(os.path.join(VOICEPACK_DIR, f'{voice}.pt'), map_location=device)
        print(f'Loaded voice: {voice} on {device}')

    return MODEL, VOICES[voice]

def update_input_visibility(choice):
    return {
        text_input: gr.update(visible=choice == "Direct Text"),
        file_input: gr.update(visible=choice == "File Upload")
    }


# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("""## Local11labs Text-to-Speech Webui
                This is a simple web interface for the Kokoro-82M text-to-speech model.""")

    # Text-to-Speech Tab
    (input_type, text_input, file_input, model_dropdown, voice_dropdown,
     speed_slider, device_dropdown, process_type, generate_button,
     audio_output, text_output, status_output) = create_text_to_speech_tab(
        models_list=MODELS_LIST,
        choices=CHOICES,
        device_options=device_options,
        update_input_visibility=update_input_visibility,
        load_model_and_voice=load_model_and_voice,
         
    )

    # Podcast Tab
    (generate_podcast_script_button, send_to_audio_input_button, podcast_script_json_output,
    podcast_dialogue_status_output, podcast_script_json_input, podcast_host_voice_assignment_inputs,
    podcast_model_dropdown, podcast_speed_slider, podcast_device_dropdown, generate_podcast_audio_button,
    podcast_audio_output, podcast_audio_status_output) = create_podcast_tab(
        models_list=MODELS_LIST,
        choices=CHOICES,
        device_options=device_options,
        kokoro_path=kokoro_path,
        load_model_and_voice=load_model_and_voice,
        
         
    )

# Run the app
if __name__ == "__main__":
    app.launch(share=True, debug=True)
