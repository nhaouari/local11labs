import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on Ampere

import numpy as np
from typing import List, Tuple, Dict, Optional
import re
from tqdm.notebook import tqdm
import gc
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import torch.cuda.amp as amp
import os
import sys
import time
from datetime import datetime
import json
import soundfile as sf
from pathlib import Path
import wave
from pydub import AudioSegment

# Ensure Kokoro is in path
kokoro_path = os.path.abspath('Kokoro-82M')
if kokoro_path not in sys.path:
    sys.path.insert(0, kokoro_path)

try:
    from kokoro import generate
except ImportError:
    print("Warning: Could not import Kokoro. Make sure the Kokoro-82M repository is cloned.")
    print("Run: git clone https://huggingface.co/hexgrad/Kokoro-82M")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Set environment variables for better CPU threading
os.environ['OMP_NUM_THREADS'] = '8'  # Adjust based on your CPU
os.environ['MKL_NUM_THREADS'] = '8'  # Adjust based on your CPU

# Increased cache size for larger chunks
CHUNK_CACHE: Dict[str, Tuple[np.ndarray, str]] = {}
MAX_CACHE_SIZE = 2000  # Doubled cache size
CACHE_MEMORY_LIMIT = 2 * 1024 * 1024 * 1024  # 2GB memory limit for cache

def get_optimal_batch_size() -> int:
    """Get aggressive batch size for maximum GPU utilization."""
    if device != 'cuda':
        return 4

    total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)

    # More aggressive memory usage (85% of free memory)
    safety_margin = 0.85
    # Further reduced per-chunk estimate
    estimated_chunk_memory = 25 * 1024 * 1024  # 25MB per chunk

    optimal_batch_size = max(4, min(32, int((free_memory * safety_margin) / estimated_chunk_memory)))
    return optimal_batch_size

def parallel_text_chunking(text: str, max_chars: int = 450) -> List[str]:
    """Split text into larger chunks for better GPU utilization."""
    # Split on paragraph boundaries first
    paragraphs = text.split('\n\n')

    # Process paragraphs in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        chunk_lists = list(executor.map(
            lambda p: split_text_into_chunks(p, max_chars),
            paragraphs
        ))

    return [chunk for chunks in chunk_lists for chunk in chunks if chunk.strip()]

def split_text_into_chunks(text: str, max_chars: int = 450) -> List[str]:
    """Split text into chunks that respect sentence boundaries."""
    text = text.strip().replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)

    if len(text) <= max_chars:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_chars:
            chunks.append(text)
            break

        split_pos = max_chars
        last_period = text[:max_chars].rfind('.')
        last_question = text[:max_chars].rfind('?')
        last_exclamation = text[:max_chars].rfind('!')

        sentence_end = max(last_period, last_question, last_exclamation)

        if sentence_end == -1:
            last_comma = text[:max_chars].rfind(',')
            last_space = text[:max_chars].rfind(' ')
            split_pos = max(last_comma, last_space)
            if split_pos == -1:
                split_pos = max_chars
        else:
            split_pos = sentence_end + 1

        chunks.append(text[:split_pos].strip())
        text = text[split_pos:].strip()

    return chunks

@lru_cache(maxsize=MAX_CACHE_SIZE)
def get_cached_chunk(chunk: str, model_hash: str, voicepack_hash: str, lang: str) -> Tuple[np.ndarray, str]:
    """Cache wrapper for chunk processing with memory management."""
    global current_cache_memory

    # Process the chunk
    audio, phonemes = process_chunk(chunk, MODEL, VOICEPACK, lang)

    # Estimate memory usage
    memory_usage = estimate_array_memory(audio)

    # Check if adding this result would exceed memory limit
    if current_cache_memory + memory_usage > CACHE_MEMORY_LIMIT:
        # Clear some cache if needed
        clear_old_cache_entries()

    current_cache_memory += memory_usage
    return audio, phonemes

def clear_old_cache_entries():
    """Clear oldest cache entries to free up memory."""
    global current_cache_memory
    while current_cache_memory > CACHE_MEMORY_LIMIT * 0.8:  # Keep 20% margin
        if not CHUNK_CACHE:
            break
        # Remove oldest entry
        _, (audio, _) = CHUNK_CACHE.popitem()
        current_cache_memory -= estimate_array_memory(audio)

class Timer:
    """Enhanced context manager for timing code blocks with real-time updates."""
    def __init__(self, description, show_start=True, verbose=True):
        self.description = description
        self.show_start = show_start
        self.verbose = verbose
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        if self.show_start and self.verbose:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting {self.description}...")
        return self

    def __exit__(self, *args):
        elapsed_time = time.time() - self.start_time
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.description}: {elapsed_time:.2f} seconds")

def process_chunk(chunk: str, model, voicepack, lang: str, output_dir: str = "output") -> Tuple[np.ndarray, str, str]:
    """Process chunk with maximum GPU utilization."""
    try:
        with Timer(f"Processing chunk of length {len(chunk)}"):
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=device=='cuda'):
                audio, phonemes = generate(model, chunk, voicepack, lang=lang)
                # Save the audio chunk
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                wav_path = os.path.join(output_dir, f'chunk_{timestamp}.wav')
                sf.write(wav_path, audio, 24000)
                return audio, phonemes, wav_path
    except Exception as e:
        print(f"Error processing chunk: {chunk[:50]}... Error: {str(e)}")
        return np.zeros(0), "", ""

def process_batch(batch: List[str], model, voicepack, lang: str, batch_num: int, total_batches: int, output_dir: str, verbose: bool = True) -> List[Tuple[np.ndarray, str, str]]:
    """Process multiple chunks in parallel with detailed logging."""
    try:
        if verbose:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting batch {batch_num}/{total_batches}")
            print(f"Batch size: {len(batch)} chunks")
            print(f"Total characters in batch: {sum(len(chunk) for chunk in batch)}")

        start_time = time.time()
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=device=='cuda'):
            results = []
            for i, chunk in enumerate(batch, 1):
                chunk_start = time.time()
                audio, phonemes, wav_path = process_chunk(chunk, model, voicepack, lang, output_dir)
                chunk_time = time.time() - chunk_start
                results.append((audio, phonemes, wav_path))
                if verbose:
                    print(f"  - Chunk {i}/{len(batch)} processed in {chunk_time:.2f}s ({len(chunk)} chars)")

        batch_time = time.time() - start_time
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch completed in {batch_time:.2f}s")
        return results
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return [(np.zeros(0), "", "")] * len(batch)

def assemble_audio_fast(audio_chunks: List[np.ndarray], chunks: List[str]) -> np.ndarray:
    """Faster audio assembly using numpy operations."""
    # Pre-calculate silence arrays
    silence_arrays = {
        'long': np.zeros(int(24000 * 0.4)),    # For sentence endings
        'medium': np.zeros(int(24000 * 0.2)),  # For commas
        'short': np.zeros(int(24000 * 0.1))    # For other breaks
    }

    # Pre-allocate final array
    total_length = sum(len(chunk) for chunk in audio_chunks)
    total_silence = len(chunks) * int(24000 * 0.4)  # Maximum possible silence
    final_audio = np.zeros(total_length + total_silence, dtype=np.float32)

    current_pos = 0
    for i, chunk in enumerate(audio_chunks):
        if len(chunk) > 0:
            final_audio[current_pos:current_pos + len(chunk)] = chunk
            current_pos += len(chunk)

            if i < len(chunks) - 1:
                # Select silence based on punctuation
                if chunks[i].strip().endswith(('.', '!', '?')):
                    silence = silence_arrays['long']
                elif chunks[i].strip().endswith(','):
                    silence = silence_arrays['medium']
                else:
                    silence = silence_arrays['short']

                final_audio[current_pos:current_pos + len(silence)] = silence
                current_pos += len(silence)

    return final_audio[:current_pos]

class TimingLogger:
    """Logger for timing and performance metrics."""
    def __init__(self):
        self.timings = {
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'chunks': [],
            'batches': [],
            'memory_usage': [],
            'total_stats': {}
        }

    def add_chunk(self, chunk_info):
        self.timings['chunks'].append(chunk_info)

    def add_batch(self, batch_info):
        self.timings['batches'].append(batch_info)

    def add_memory_usage(self, memory_info):
        self.timings['memory_usage'].append(memory_info)

    def set_total_stats(self, stats):
        self.timings['total_stats'] = stats

    def save(self, output_dir: str):
        """Save timing information to a JSON file."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(output_dir) / f'tts_timing_log_{timestamp}.json'
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.timings, f, indent=2)
        return log_file

def save_audio_file(audio_data: np.ndarray, sample_rate: int, output_dir: str) -> str:
    """Save audio data to both WAV and MP3 formats."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save WAV
    wav_path = str(Path(output_dir) / f'tts_output_{timestamp}.wav')
    sf.write(wav_path, audio_data, sample_rate)

    return wav_path

def generate_long_text_optimized(
    model,
    text: str,
    voicepack,
    lang: str,
    output_dir: str = "output",
    verbose: bool = False
) -> Tuple[np.ndarray, str, str]:
    """Generate audio with maximum GPU utilization and save results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize timing logger
    logger = TimingLogger()
    start_time = time.time()

    if verbose:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting text-to-speech generation...")
        print(f"Total text length: {len(text)} characters")

    # Initial cleanup only
    if device == 'cuda':
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.get_device_properties(0).total_memory/1024**2
        if verbose:
            print(f"GPU Memory cleared - Available: {initial_memory:.1f}MB")
        logger.add_memory_usage({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'event': 'initial',
            'available_memory': initial_memory
        })

    # Get larger chunks
    with Timer("Text chunking", verbose=verbose):
        chunks = parallel_text_chunking(text)
        if verbose:
            print(f"Split into {len(chunks)} chunks")
            print("Chunk sizes:", [len(chunk) for chunk in chunks])

    if len(chunks) == 1:
        try:
            return process_chunk(chunks[0], model, voicepack, lang, output_dir)
        except Exception as e:
            print(f"Error processing single chunk: {str(e)}")
            # Return silent audio as fallback
            silence = np.zeros(int(24000 * 0.5))  # 0.5 seconds of silence
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fallback_path = os.path.join(output_dir, f'fallback_{timestamp}.wav')
            sf.write(fallback_path, silence, 24000)
            return silence, "", fallback_path

    # Get aggressive batch size
    batch_size = get_optimal_batch_size()
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    if verbose:
        print(f"\nUsing batch size: {batch_size} for {len(chunks)} chunks ({total_batches} batches)")

    audio_chunks = []
    phonemes = []
    wav_paths = []

    # Process batches with detailed tracking
    batch_times = []
    total_chars_processed = 0

    progress_bar = tqdm(total=len(chunks), desc="Processing text chunks", disable=not verbose)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1

        # Process entire batch
        batch_start = time.time()
        batch_results = process_batch(batch, model, voicepack, lang, batch_num, total_batches, output_dir, verbose)

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        # Log batch information
        chars_in_batch = sum(len(chunk) for chunk in batch)
        total_chars_processed += chars_in_batch
        elapsed = time.time() - start_time
        chars_per_second = total_chars_processed / elapsed if elapsed > 0 else 0

        logger.add_batch({
            'batch_number': batch_num,
            'batch_size': len(batch),
            'chars_processed': chars_in_batch,
            'processing_time': batch_time,
            'chars_per_second': chars_per_second
        })

        if verbose:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Progress Update:")
            print(f"  - Chars processed: {total_chars_processed}/{len(text)} ({total_chars_processed/len(text)*100:.1f}%)")
            print(f"  - Processing speed: {chars_per_second:.1f} chars/second")
            if chars_per_second > 0:
                print(f"  - Estimated time remaining: {(len(text) - total_chars_processed) / chars_per_second:.1f}s")

        # Memory stats
        if device == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.add_memory_usage({
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'batch_number': batch_num,
                'allocated_memory': allocated,
                'reserved_memory': reserved
            })
            if verbose:
                print(f"  - GPU Memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")

        progress_bar.update(len(batch))

        batch_audio, batch_phonemes, batch_paths = zip(*batch_results)
        audio_chunks.extend(batch_audio)
        phonemes.extend(batch_phonemes)
        wav_paths.extend(batch_paths)

    progress_bar.close()

    # Check if we have any valid audio chunks
    valid_audio_chunks = [chunk for chunk in audio_chunks if len(chunk) > 0]
    if not valid_audio_chunks:
        print("Warning: No valid audio chunks generated")
        silence = np.zeros(int(24000 * 0.5))  # 0.5 seconds of silence
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fallback_path = os.path.join(output_dir, f'fallback_{timestamp}.wav')
        sf.write(fallback_path, silence, 24000)
        return silence, "", fallback_path

    # Assemble final audio
    with Timer("Audio assembly", verbose=verbose):
        final_audio = assemble_audio_fast(valid_audio_chunks, chunks)

    total_time = time.time() - start_time
    chars_per_second = len(text)/total_time if total_time > 0 else 0
    audio_length = len(final_audio)/24000 if len(final_audio) > 0 else 0.1  # Avoid division by zero

    logger.set_total_stats({
        'total_chunks': len(chunks),
        'total_batches': total_batches,
        'avg_batch_time': sum(batch_times)/len(batch_times) if batch_times else 0,
        'max_batch_time': max(batch_times) if batch_times else 0,
        'min_batch_time': min(batch_times) if batch_times else 0,
        'total_chars': len(text),
        'total_time': total_time,
        'chars_per_second': chars_per_second,
        'audio_length': audio_length
    })

    # Always show the final summary, even in non-verbose mode
    print(f"\n{'=' * 40}")
    print("Text-to-Speech Generation Summary")
    print(f"{'=' * 40}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Processing speed: {chars_per_second:.1f} characters/second")
    print(f"Text length: {len(text)} characters")
    print(f"Audio length: {audio_length:.1f} seconds ({audio_length/60:.1f} minutes)")
    if audio_length > 0:
        print(f"Average speaking rate: {len(text)/audio_length:.1f} characters/second")
    print(f"{'=' * 40}")

    # Save results
    if verbose:
        print("\nSaving results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_wav_path = os.path.join(output_dir, f'tts_output_{timestamp}.wav')
    try:
        sf.write(final_wav_path, final_audio, 24000)
        print(f"Audio saved successfully to: {final_wav_path}")
    except Exception as e:
        print(f"Error saving audio: {str(e)}")
        # Create a fallback path in the current directory
        final_wav_path = f'tts_output_{timestamp}.wav'
        sf.write(final_wav_path, final_audio, 24000)
        print(f"Audio saved to fallback location: {final_wav_path}")

    try:
        log_path = logger.save(output_dir)
        print(f"Log saved to: {log_path}")
    except Exception as e:
        print(f"Error saving log: {str(e)}")

    return final_audio, '\n'.join(phonemes), final_wav_path

def convert_to_mp3(wav_path: str) -> str:
    """Convert WAV file to MP3 format with good quality settings."""
    try:
        # Create MP3 path
        mp3_path = wav_path.rsplit('.', 1)[0] + '.mp3'

        # Load WAV and convert to MP3
        audio = AudioSegment.from_wav(wav_path)

        # Normalize and enhance
        audio = audio.normalize()

        # Export as MP3 with good quality settings
        audio.export(
            mp3_path,
            format="mp3",
            bitrate="256k",
            parameters=[
                "-ac", "2",  # Stereo output
                "-ar", "48000",  # Higher sample rate
                "-q:a", "0"  # Highest quality
            ]
        )

        print(f"Converted to MP3: {mp3_path}")
        return mp3_path
    except Exception as e:
        print(f"Error converting to MP3: {str(e)}")
        return wav_path

def load_voicepack(voice_name: str, path: str = None):
    """Load a specific voicepack.
    
    Args:
        voice_name: Name of the voice to load
        path: Optional absolute path to voices directory. If not provided, uses relative 'voices' directory.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Determine voice file path
    if path:
        voice_path = os.path.join(path, f'{voice_name}.pt')
    else:
        voice_path = f'voices/{voice_name}.pt'

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=device=='cuda'):
        voicepack = torch.load(voice_path, weights_only=True)
        if isinstance(voicepack, torch.Tensor):
            voicepack = voicepack.detach().to(device)
            if device == 'cuda':
                voicepack = voicepack.half()
        elif isinstance(voicepack, dict):
            for k in voicepack.keys():
                if isinstance(voicepack[k], torch.Tensor):
                    voicepack[k] = voicepack[k].detach().to(device)
                    if device == 'cuda':
                        voicepack[k] = voicepack[k].half()
    return voicepack

def load_content(file_path="/content/content.txt"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content 