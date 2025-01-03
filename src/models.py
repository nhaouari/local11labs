import os
import torch
import sys
from pathlib import Path

def build_model(model_path="Kokoro-82M"):
    """Build and load the Kokoro model.
    
    Args:
        model_path (str): Path to the Kokoro model directory or repository name
    """
    try:
        # If model_path doesn't exist, clone it from HuggingFace
        if not os.path.exists(model_path):
            print(f"Cloning Kokoro model from HuggingFace...")
            os.system(f"git clone https://huggingface.co/hexgrad/Kokoro-82M {model_path}")

        # Add Kokoro directory to Python path
        kokoro_path = os.path.abspath(model_path)
        if kokoro_path not in sys.path:
            sys.path.append(kokoro_path)

        # Import kokoro after ensuring the model exists
        from kokoro import load_model

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading model on {device}...")

        # Load the model with optimizations
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=device=='cuda'):
            model = load_model(model_path)
            if device == 'cuda':
                model = model.half()  # Use half precision on GPU
            model = model.to(device)
            model.eval()  # Set to evaluation mode

        return model

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise 