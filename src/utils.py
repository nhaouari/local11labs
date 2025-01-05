import torch
import tempfile
import soundfile as sf
from long_speech_generation import generate_long_text_optimized
from kokoro import generate

def read_file_content(file):
    if file is None:
        return ""
    with open(file.name, 'r', encoding='utf-8') as f:
        return f.read()

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
        return (24000, audio), phonemes, wav_path
    except Exception as e:
        print(f"Error in long text processing: {str(e)}")
        return None, str(e), None

def generate_audio_enhanced(
    text,
    model_name,
    voice_name,
    speed,
    selected_device,
    load_model_and_voice,
    is_long_text=False,
    output_dir="output"
):
    """Enhanced audio generation with support for long text and podcasts."""
    if not text.strip():
        return (None, "", None)

    # Load model and voice
    model, voice_data = load_model_and_voice(selected_device, model_name, voice_name)

    try:
        if is_long_text:
            return process_long_text(
                text=text,
                model=model,
                voice_data=voice_data,
                lang=voice_name[0],
                speed=speed,
                output_dir=output_dir
            )
        else:
            audio, phonemes = generate(
                model=model,
                text=text,
                voicepack=voice_data,
                speed=speed,
                lang=voice_name[0]
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                sf.write(temp_file.name, audio, 24000)
                return (24000, audio), phonemes, temp_file.name
    except Exception as e:
        print(f"Error in audio generation: {str(e)}")
        return None, str(e), None
