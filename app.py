import torch
import gradio as gr
import os
import re
import sys
from src.long_speech_generation import generate_long_text_optimized  
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


VOICEPACK_DIR = os.path.join(kokoro_path, "voices")  # Ensure this directory exists in your local_model_path

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
    try :
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
        # print(f"Error: {e}")
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
        VOICES[voice] = torch.load(os.path.join(VOICEPACK_DIR, f'{voice}.pt'), weights_only=True).to(device)
        print(f'Loaded voice: {voice} on {device}')

    return MODEL, VOICES[voice]

def process_long_text(text, model, voice_data, lang, speed, output_dir="output"):
    """Process longer text with optimized generation."""
    try:
        audio, phonemes, wav_path = generate_long_text_optimized(
            model=model,
            text=text,
            voicepack=voice_data,
            lang=lang,
            output_dir=output_dir,
            verbose=True
        )
        return (SAMPLE_RATE, audio), phonemes, wav_path
    except Exception as e:
        print(f"Error in long text processing: {str(e)}")
        return None, str(e), None

def generate_audio_enhanced(
    text, 
    model_name, 
    voice_name, 
    speed, 
    selected_device, 
    is_long_text=False,
    output_dir="output"
):
    """Enhanced audio generation with support for long text and podcasts."""
    if not text.strip():
        return (None, "", None)
    
    # Extract voice and model information
    voice = voice_name[1] if isinstance(voice_name, tuple) else voice_name
    model_path = model_name[1] if isinstance(model_name, tuple) else model_name
    
    # Load model and voice
    model, voice_data = load_model_and_voice(selected_device, model_path, voice)
    
    try:
        if is_long_text:
            # Use optimized processing for long text
            return process_long_text(
                text=text,
                model=model,
                voice_data=voice_data,
                lang=voice[0],
                speed=speed,
                output_dir=output_dir
            )
        else:
            # Use standard processing for short text
            audio, phonemes = generate(
                model=model,
                text=text,
                voicepack=voice_data,
                speed=speed,
                lang=voice[0]
            )
            
            # Save the audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                import soundfile as sf
                sf.write(temp_file.name, audio, SAMPLE_RATE)
                return (SAMPLE_RATE, audio), phonemes, temp_file.name
    except Exception as e:
        print(f"Error in audio generation: {str(e)}")
        return None, str(e), None


#------
def update_input_visibility(choice):
    return {
        text_input: choice == "Direct Text",
        file_input: choice == "File Upload"
    }
    
def read_file_content(file):
    if file is None:
        return ""
    with open(file.name, 'r', encoding='utf-8') as f:
        return f.read()

def generate_wrapper(text, file, input_type, model_name, voice_name, speed, selected_device, is_long_text):
    try:
        # Determine input text
        if input_type == "File Upload" and file is not None:
            text = read_file_content(file)
        
        # Update status
        gr.update(value="Processing...")
        
        # Generate audio
        audio_result, phonemes, wav_path = generate_audio_enhanced(
            text=text,
            model_name=model_name,
            voice_name=voice_name,
            speed=speed,
            selected_device=selected_device,
            is_long_text=is_long_text
        )
        
        status = "Generation complete!"
        return audio_result, phonemes, status
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return None, error_msg, error_msg
    
# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("""## Local11labs Text-to-Speech Webui
                This is a simple web interface for the Kokoro-82M text-to-speech model.""")
    with gr.Row():
        with gr.Column():
            # Text input options
            input_type = gr.Radio(
                choices=["Direct Text", "File Upload"],
                value="Direct Text",
                label="Input Type"
            )
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text here...",
                visible=True
            )
            file_input = gr.File(
                label="Upload Text File",
                visible=False
            )
            
            # Model and voice selection
            model_dropdown = gr.Dropdown(
                list(MODELS_LIST.items()),
                label="Model",
                # value=("v0_19-full-fp32", "kokoro-v0_19.pth")
                value=os.path.join(kokoro_path, "fp16/kokoro-v0_19-half.pth"),
            )
            voice_dropdown = gr.Dropdown(
                list(CHOICES.items()),
                label="Voice",
                value="af"
                # value=("🇺🇸 🚺 American Female ⭐", "af")
            )
            
            # Generation settings
            speed_slider = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Speed"
            )
            device_dropdown = gr.Dropdown(
                device_options,
                label="Device",
                value="auto"
            )
            process_type = gr.Checkbox(
                label="Process as Long Text",
                value=False
            )
            generate_button = gr.Button("Generate")
            
        with gr.Column():
            # Output components
            audio_output = gr.Audio(label="Output Audio")
            text_output = gr.Textbox(label="Output Phonemes")
            status_output = gr.Textbox(label="Status", value="Ready")
            
    
    # Event handlers
    input_type.change(
        update_input_visibility,
        inputs=[input_type],
        outputs=[text_input, file_input]
    )
    
    generate_button.click(
        generate_wrapper,
        inputs=[
            text_input,
            file_input,
            input_type,
            model_dropdown,
            voice_dropdown,
            speed_slider,
            device_dropdown,
            process_type
        ],
        outputs=[
            audio_output,
            text_output,
            status_output
        ]
    )

# Run the app
if __name__ == "__main__":
    app.launch(share=True, debug=True)