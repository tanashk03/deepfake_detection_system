"""
Luminark Advanced Training Pipeline

Train the advanced multi-modal detectors (Spatial, Temporal, Frequency).
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

from core.models.spatial import create_spatial_detector
from core.models.temporal import create_temporal_detector
from core.models.frequency import create_frequency_detector
from core.models.physiological import create_physiological_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LuminarkTrain")

class DeepfakeDataset(Dataset):
    """Dataset loader for Real/Fake video folder structure."""
    
    def __init__(self, root_dir, modality='spatial', transform=None):
        self.root_dir = Path(root_dir)
        self.modality = modality
        self.samples = []
        
        # Expect 'real' and 'fake' subfolders
        # 0=Fake, 1=Real to match standard logic
        for label, class_name in enumerate(['fake', 'real']): 
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            for fname in class_dir.glob("*"):
                if fname.suffix.lower() in ['.mp4', '.avi', '.mov', '.webm']:
                    self.samples.append((str(fname), label))
                    
        logger.info(f"Loaded {len(self.samples)} {modality} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self._load_data(path, self.modality), label

    def _load_data(self, path, modality):
        cap = cv2.VideoCapture(path)
        
        if modality == 'spatial' or modality == 'frequency':
            # Single frame (middle)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return np.zeros((3, 224, 224), dtype=np.float32)
                
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            return torch.FloatTensor(frame).permute(2, 0, 1) # CHW
            
        elif modality == 'temporal' or modality == 'physiological':
            # Sequence of 16 frames
            frames = []
            while len(frames) < 16:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            cap.release()
            
            # Pad
            while len(frames) < 16:
                frames.append(np.zeros((224, 224, 3), dtype=np.float32))
                
            frames = np.array(frames)
            # Both temporal and physiological expect (N, C, H, W) or (N, T, C, H, W) logic
            # Temporal (VideoMAE): (B, C, T, H, W) usually, but my wrapper might handle it.
            # Physiological: (B, T, C, H, W) internal extractors handle it?
            # Let's check logic. PhysiologicalDetector expects (B, T, C, H, W) implicitly via extract_rppg(video_frames).
            # video_frames in extract_rppg is (batch, seq_len, 3, 224, 224)? No, extract_rppg uses [b,t].
            
            tensor_frames = torch.FloatTensor(frames) # (16, 224, 224, 3)
            
            if modality == 'physiological':
                 # PhysiologicalDetector expects (B, T, 3, H, W) based on extract_rppg logic where it does:
                 # frame = video_frames[b, t].cpu().numpy().transpose(1, 2, 0)
                 # Wait, if input is (B, T, 3, H, W), then [b,t] is (3,H,W). Transpose(1,2,0) makes it (H,W,3). Correct.
                 return tensor_frames.permute(0, 3, 1, 2) # (T, C, H, W)
                 
            return tensor_frames.permute(0, 3, 1, 2) # (T, C, H, W) - VideoMAE wrapper expects similar?
            
            # Actually VideoMAE usually expects (C, T, H, W).
            # TemporalDetector forward: outputs = self.video_transformer(video_frames)
            # huggingface VideoMAE expects (batch, num_frames, num_channels, height, width) OR (batch, num_channels, num_frames, height, width).
            # Default is (batch, num_frames, num_channels, height, width).
            # Let's standardise to (T, C, H, W) here and let collate make it (B, T, C, H, W).

def train_modality(modality, data_dir, epochs=5, batch_size=4, device='cpu'):
    logger.info(f"Starting training for {modality}...")
    
    if modality == 'spatial':
        model = create_spatial_detector(pretrained=True).to(device)
    elif modality == 'temporal':
        model = create_temporal_detector().to(device)
    elif modality == 'frequency':
        model = create_frequency_detector().to(device)
    elif modality == 'physiological':
        model = create_physiological_detector().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    dataset = DeepfakeDataset(data_dir, modality=modality)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': total_loss / (total/batch_size + 1), 'acc': 100 * correct / total})
            
        logger.info(f"Epoch {epoch+1} complete. Acc: {100*correct/total:.2f}%")
        
        # Save every epoch
        save_path = f"models/{modality}_finetuned.pt"
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Progress saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=['spatial', 'temporal', 'frequency', 'physiological'], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default='cpu')
    args = parser.parse_args()
    
    train_modality(args.modality, args.data, epochs=args.epochs, batch_size=args.batch_size, device=args.device)

