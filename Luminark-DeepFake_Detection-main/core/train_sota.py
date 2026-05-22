"""
Luminark SOTA Training Pipeline

Fine-tune the SOTA backbone models (WavLM, VideoMAE) on your own dataset.
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from core.models.wavlm import WavLMAnalyzer
from core.models.videomae import VideoMAEAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LuminarkTrain")

class DeepfakeDataset(Dataset):
    """Dataset loader for Real/Fake video folder structure."""
    
    def __init__(self, root_dir, modality='video', transform=None):
        self.root_dir = Path(root_dir)
        self.modality = modality
        self.samples = []
        
        # Expect 'real' and 'fake' subfolders
        for label, class_name in enumerate(['fake', 'real']):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            for fname in class_dir.glob("*"):
                if fname.suffix.lower() in ['.mp4', '.avi', '.mov', '.webm', '.wav', '.mp3']:
                    self.samples.append((str(fname), label))
                    
        logger.info(f"Loaded {len(self.samples)} {modality} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        if self.modality == 'video':
            return self._load_video(path), label
        else:
            return self._load_audio(path), label

    def _load_video(self, path):
        # Sample frames
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < 16:
            ret, frame = cap.read()
            if not ret: break
            # Resize strictly to 224x224 to ensure batch stack works
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()
        
        if not frames:
            return np.zeros((16, 224, 224, 3), dtype=np.uint8)
            
        # Pad if short
        while len(frames) < 16:
             frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
        frames = np.array(frames)
        return frames

    def _load_audio(self, path):
        import librosa
        audio, _ = librosa.load(path, sr=16000, duration=5.0)
        # Pad if short
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
        return audio

def train_modality(modality, data_dir, epochs=5, batch_size=4, device='cpu'):
    logger.info(f"Starting training for {modality}...")
    
    # Initialize Model
    if modality == 'video':
        analyzer = VideoMAEAnalyzer(device=device)
    else:
        analyzer = WavLMAnalyzer(device=device)
        
    analyzer.load_model()
    model = analyzer.backbone
    classifier = analyzer.classifier
    
    # Freeze backbone initially
    for param in model.parameters():
        param.requires_grad = False
        
    # Optimizer for head
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # DataLoader
    dataset = DeepfakeDataset(data_dir, modality=modality)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    classifier.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs = inputs  # Numpy array
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            batch_embeddings = []
            
            # Process batch items one by one (simplest integration with existing preprocess)
            for i in range(len(inputs)):
                item_input = inputs[i]
                if isinstance(item_input, torch.Tensor):
                    item_input = item_input.numpy()
                
                # Preprocess -> Features key
                if modality == 'video':
                    # shape (16, 224, 224, 3) -> model inputs
                    model_inputs = analyzer.preprocess(item_input) # Returns dict on device
                    with torch.no_grad():
                        outputs = model(**model_inputs)
                        # Mean pool: (1, seq, hidden) -> (1, hidden)
                        embedding = outputs.last_hidden_state.mean(dim=1)
                else:
                    # Audio
                    model_inputs = analyzer.preprocess(item_input) # Returns tensor on device
                    with torch.no_grad():
                        outputs = model(model_inputs)
                        embedding = outputs.last_hidden_state.mean(dim=1)
                        
                batch_embeddings.append(embedding)
            
            # Stack embeddings: (Batch, Hidden)
            batch_embeddings = torch.cat(batch_embeddings, dim=0)
            
            # Classifier Forward
            logits = classifier(batch_embeddings)
            
            # --- Backward Pass ---
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Acc stats
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': total_loss / (total/batch_size + 1), 'acc': 100 * correct / total})
            
        logger.info(f"Epoch {epoch+1} complete.")
        
    # Save
    save_path = f"models/{modality}_finetuned.pt"
    os.makedirs("models", exist_ok=True)
    torch.save(classifier.state_dict(), save_path)
    logger.info(f"Saved fine-tuned model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=['video', 'audio'], required=True)
    parser.add_argument("--data", required=True, help="Path to folder containing 'real' and 'fake' subfolders")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    train_modality(args.modality, args.data, epochs=args.epochs)
