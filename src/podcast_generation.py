import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
import re 
import numpy as np
import soundfile as sf
from long_speech_generation import load_voicepack,generate_long_text_optimized
from models import build_model

# Cache for loaded voicepacks
VOICEPACKS = {}

def preprocess_text(text: str) -> str:
    """Preprocess text to handle common issues and ensure proper line handling."""
    # Remove any newlines and convert to single line
    text = text.replace('\n', ' ')

    # Handle ellipsis properly (preserve them)
    text = text.replace('...', '###ELLIPSIS###')

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Handle quotes properly
    text = text.replace('"', "'")

    # Add proper spacing around punctuation, but not inside numbers or quotes
    text = re.sub(r'(?<!\d)([.,!?])(?!\d)', r' \1 ', text)

    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Restore ellipsis
    text = text.replace('###ELLIPSIS###', '...')

    # Ensure proper sentence endings
    if not text.strip().endswith(('.', '!', '?', '...')):
        text = text.strip() + '.'

    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)

    # Ensure single space after punctuation
    text = re.sub(r'([.,!?])\s*', r'\1 ', text)

    # Handle numbers with proper spacing
    text = re.sub(r'(\d+)\s*([.,])\s*(\d+)', r'\1\2\3', text)

    return text.strip()

def generate_audio(entry, index, host_voice_map,output_dir,model,path):
    """Generate audio for a single dialogue entry with optimized speed."""
    try:
        host = entry["host"]
        text = entry["dialogue"]

        # Preprocess the text
        text = preprocess_text(text)

        # Default voice
        VOICE_NAME='am_michael'

        # Get voice settings for this host
        voice_settings = host_voice_map.get(host, {
            'voice':VOICE_NAME,
            'lang': VOICE_NAME[0]
        })

        # Split text into sentences more carefully
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Add pauses between sentences
        text_with_pauses = ' ... '.join(sentences)

        # Load the specific voicepack for this host (cached)
        voicepack = VOICEPACKS.get(voice_settings['voice'])
        if voicepack is None:
            try:
                voicepack = load_voicepack(voice_settings['voice'],path)
                VOICEPACKS[voice_settings['voice']] = voicepack
            except Exception as e:
                print(f"Error loading voicepack for {host}: {str(e)}")
                voice_settings = {'voice': VOICE_NAME, 'lang': VOICE_NAME[0]}
                voicepack = VOICEPACKS.get(VOICE_NAME) or load_voicepack(VOICE_NAME,path)
                VOICEPACKS[VOICE_NAME] = voicepack

        print(f"\nGenerating audio for {host} (Voice: {voice_settings['voice']}, Lang: {voice_settings['lang']})...")
        print(f"Text to process: {text_with_pauses}")

        # Process text in smaller chunks while preserving sentence boundaries
        max_chunk_length = 400  # Reduced chunk size for better handling
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > max_chunk_length and current_chunk:
                chunks.append(' ... '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)

        if current_chunk:
            chunks.append(' ... '.join(current_chunk))

        # Process each chunk
        all_audio = []
        all_phonemes = []

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} of length {len(chunk)}...")
            try:
                chunk_audio, chunk_phonemes, _ = generate_long_text_optimized(
                    model=model,
                    text=chunk,
                    voicepack=voicepack,
                    lang=voice_settings['lang'],
                    output_dir=output_dir,
                    verbose=False
                )
                if len(chunk_audio) > 0:
                    all_audio.append(chunk_audio)
                    all_phonemes.append(chunk_phonemes)
            except Exception as e:
                print(f"Error processing chunk: {chunk}\nError: {str(e)}")
                continue

        # Concatenate all audio chunks with proper silence
        if all_audio:
            silence = np.zeros(int(24000 * 0.3))  # 0.3 seconds silence between chunks
            final_audio = np.concatenate([np.concatenate([chunk, silence]) for chunk in all_audio[:-1]] + [all_audio[-1]])
        else:
            final_audio = np.array([])

        # Create output path
        new_wav_path = os.path.join(output_dir, f"{index:04d}_{host}_{voice_settings['voice']}.wav")

        # Normalize and save audio
        if len(final_audio) > 0:
            if np.max(np.abs(final_audio)) > 0:
                final_audio = final_audio / np.max(np.abs(final_audio)) * 0.95
            sf.write(new_wav_path, final_audio, 24000)
            print(f"Generated audio for {host}: {new_wav_path}")
        else:
            print(f"Warning: Empty audio generated for text: {text[:100]}...")
            silence = np.zeros(int(24000 * 1.0))
            sf.write(new_wav_path, silence, 24000)
            print(f"Created silence for {host}: {new_wav_path}")

        return (index, new_wav_path)

    except Exception as e:
        print(f"Error in generate_audio for {host} at index {index}: {str(e)}")
        fallback_path = os.path.join(output_dir, f"{index:04d}_error_fallback.wav")
        silence = np.zeros(int(24000 * 1.0))
        sf.write(fallback_path, silence, 24000)
        return (index, fallback_path)

def merge_audio_files(audio_files, output_file="merged_podcast.mp3"):
    """Merge audio files with optimized processing."""
    try:
        print("\nMerging audio files...")
        merged_audio = None
        crossfade_duration = 100  # Reduced for faster processing

        # Sort audio files by index to ensure correct order
        audio_files.sort(key=lambda x: x[0])

        # Create a list of valid files with proper ordering
        valid_files = []
        for i, (index, file) in enumerate(audio_files):
            if file and os.path.exists(file):
                valid_files.append((index, file))
            else:
                print(f"Warning: Missing audio file at index {index}")
                fallback_path = os.path.join(output_dir, f"{index:04d}_missing_fallback.wav")
                silence = np.zeros(int(24000 * 0.5))
                sf.write(fallback_path, silence, 24000)
                valid_files.append((index, fallback_path))

        # Process files in order with basic transitions
        for i, (index, file) in enumerate(valid_files):
            print(f"Processing file {i+1}/{len(valid_files)}: {os.path.basename(file)} (Index: {index})")
            try:
                segment = AudioSegment.from_wav(file)
                segment = segment.normalize()

                # Simple silence between segments
                silence_duration = 500
                silence = AudioSegment.silent(duration=silence_duration)

                if merged_audio is None:
                    merged_audio = segment
                else:
                    merged_audio = merged_audio.append(segment, crossfade=crossfade_duration)

                if i < len(valid_files) - 1:
                    merged_audio = merged_audio + silence

            except Exception as e:
                print(f"Warning: Error processing file {file}: {str(e)}")
                continue

        if merged_audio is None:
            raise Exception("No valid audio files were processed")

        # Basic audio enhancement
        merged_audio = merged_audio.normalize()

        # Export with good quality but faster processing
        print(f"\nExporting final podcast to {output_file}...")
        merged_audio.export(
            output_file,
            format="mp3",
            bitrate="192k",
            parameters=["-ac", "2"]
        )

        duration_minutes = len(merged_audio)/(1000 * 60)
        print(f"Successfully created podcast: {output_file}")
        print(f"Duration: {duration_minutes:.1f} minutes")

    except Exception as e:
        print(f"Error merging audio files: {str(e)}")
        silence = AudioSegment.silent(duration=1000)
        silence.export(output_file, format="mp3", bitrate="192k", parameters=["-ac", "2"])
        print(f"Created fallback podcast: {output_file}")
        raise

def clean_output_folder(folder):
    """Remove all files in the output folder."""
    print(f"Cleaning output folder: {folder}")
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or link
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                os.rmdir(file_path)  # Remove empty directories if any
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}") 