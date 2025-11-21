"""
Training script for Temporal Retrieval Encoder

Implements three temporal learning objectives:
1. Semantic contrastive loss - preserve content meaning
2. Temporal proximity loss - nearby events should have similar embeddings
3. Time-window membership loss - teach concepts like "last 6 months"

Usage:
    python train_temporal_encoder.py \\
        --data_path benchmarks/LongMemEval/data/longmemeval_s_cleaned.json \\
        --base_model Alibaba-NLP/gte-base-en-v1.5 \\
        --epochs 10
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm

from temporal_retriever import TemporalEncoder, timestamp_to_seconds


class TemporalConversationDataset(Dataset):
    """
    Dataset for training temporal encoders on LongMemEval data.

    Creates training triplets:
    - Anchor: A conversation session
    - Positive: Semantically similar + temporally close
    - Negative: Dissimilar or temporally distant
    """

    def __init__(
        self,
        data_path: str,
        max_sessions_per_question: int = 50,
        temporal_window_days: int = 7
    ):
        """
        Args:
            data_path: Path to LongMemEval JSON file
            max_sessions_per_question: Limit sessions to process
            temporal_window_days: Days within which sessions are "temporally close"
        """
        self.data = json.load(open(data_path))
        self.temporal_window = temporal_window_days * 86400  # Convert to seconds
        self.max_sessions = max_sessions_per_question

        # Build session pool
        self.sessions = []
        for entry in self.data:
            for sess_idx, (session, date, sess_id) in enumerate(zip(
                entry['haystack_sessions'][:self.max_sessions],
                entry['haystack_dates'][:self.max_sessions],
                entry['haystack_session_ids'][:self.max_sessions]
            )):
                # Convert session to text
                text = self._session_to_text(session)
                timestamp = timestamp_to_seconds(date)

                # Check if answer session
                is_answer = sess_id in entry.get('answer_session_ids', [])

                self.sessions.append({
                    'text': text,
                    'timestamp': timestamp,
                    'question_id': entry['question_id'],
                    'session_id': sess_id,
                    'is_answer': is_answer,
                    'question_type': entry['question_type']
                })

        print(f"Loaded {len(self.sessions)} sessions from {len(self.data)} questions")

    def _session_to_text(self, session: List[Dict]) -> str:
        """Convert session turns to text."""
        texts = []
        for turn in session:
            texts.append(f"{turn['role']}: {turn['content']}")
        return " ".join(texts)

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get anchor session and find positive/negative examples.

        Returns:
            Dict with anchor, positive, negative sessions and temporal info
        """
        anchor = self.sessions[idx]

        # Find positive: same question, temporally close
        positives = [
            s for s in self.sessions
            if s['question_id'] == anchor['question_id']
            and s['session_id'] != anchor['session_id']
            and abs(s['timestamp'] - anchor['timestamp']) < self.temporal_window
        ]

        # Find hard negative: same question but temporally distant
        hard_negatives = [
            s for s in self.sessions
            if s['question_id'] == anchor['question_id']
            and s['session_id'] != anchor['session_id']
            and abs(s['timestamp'] - anchor['timestamp']) >= self.temporal_window
        ]

        # Find easy negative: different question
        easy_negatives = [
            s for s in self.sessions
            if s['question_id'] != anchor['question_id']
        ]

        # Sample positive and negative
        if positives:
            positive = np.random.choice(positives)
        else:
            # Fallback: use anchor itself
            positive = anchor

        if hard_negatives:
            negative = np.random.choice(hard_negatives)
        elif easy_negatives:
            negative = np.random.choice(easy_negatives)
        else:
            # Fallback: use random session
            negative = self.sessions[np.random.randint(len(self.sessions))]

        return {
            'anchor_text': anchor['text'],
            'anchor_time': anchor['timestamp'],
            'positive_text': positive['text'],
            'positive_time': positive['timestamp'],
            'negative_text': negative['text'],
            'negative_time': negative['timestamp'],
            'time_diff_pos': abs(positive['timestamp'] - anchor['timestamp']),
            'time_diff_neg': abs(negative['timestamp'] - anchor['timestamp'])
        }


class TemporalLosses(nn.Module):
    """
    Combined loss functions for temporal encoding.

    1. Semantic Contrastive Loss: InfoNCE-style contrastive learning
    2. Temporal Proximity Loss: Closer in time → closer in embedding space
    3. Time-Window Membership Loss: Teach temporal bucket concepts
    """

    def __init__(
        self,
        temperature: float = 0.07,
        temporal_weight: float = 0.3,
        window_weight: float = 0.2
    ):
        """
        Args:
            temperature: Temperature for contrastive loss
            temporal_weight: Weight for temporal proximity loss
            window_weight: Weight for window membership loss
        """
        super().__init__()
        self.temperature = temperature
        self.temporal_weight = temporal_weight
        self.window_weight = window_weight

    def semantic_contrastive_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss.

        Pull anchor closer to positive, push away from negative.

        Args:
            anchor, positive, negative: Embeddings [batch, dim]

        Returns:
            Contrastive loss scalar
        """
        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        # Positive similarity
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature

        # Negative similarity
        neg_sim = (anchor * negative).sum(dim=-1) / self.temperature

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        return loss

    def temporal_proximity_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        time_diff_pos: torch.Tensor,
        time_diff_neg: torch.Tensor
    ) -> torch.Tensor:
        """
        Temporal proximity loss.

        Embedding distance should correlate with temporal distance.
        Positive (closer in time) should be closer in embedding space than negative.

        Args:
            anchor, positive, negative: Embeddings [batch, dim]
            time_diff_pos, time_diff_neg: Time differences in seconds [batch]

        Returns:
            Temporal loss scalar
        """
        # Embedding distances
        dist_pos = F.pairwise_distance(anchor, positive)
        dist_neg = F.pairwise_distance(anchor, negative)

        # Normalize time differences to [0, 1] range for stability
        time_diff_pos = torch.log(time_diff_pos + 1) / 10.0
        time_diff_neg = torch.log(time_diff_neg + 1) / 10.0

        # Loss: embedding distance should match temporal distance ordering
        # If time_diff_pos < time_diff_neg, then dist_pos should be < dist_neg
        margin = 0.5
        loss = F.relu(dist_pos - dist_neg + margin * (time_diff_neg - time_diff_pos))

        return loss.mean()

    def time_window_membership_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        time_diff_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Time-window membership loss.

        Teach the model to recognize temporal buckets:
        - Same day (< 1 day)
        - Same week (< 7 days)
        - Same month (< 30 days)
        - etc.

        Args:
            anchor, positive: Embeddings [batch, dim]
            time_diff_pos: Time difference in seconds [batch]

        Returns:
            Window loss scalar
        """
        # Define windows (in seconds)
        windows = torch.tensor([
            86400,        # 1 day
            7 * 86400,    # 1 week
            30 * 86400,   # 1 month
            365 * 86400   # 1 year
        ], device=time_diff_pos.device)

        # Determine which window each pair belongs to
        # Shape: [batch, num_windows]
        in_window = time_diff_pos.unsqueeze(1) < windows.unsqueeze(0)

        # Similarity should be higher for closer windows
        similarity = F.cosine_similarity(anchor, positive)

        # Target: higher similarity for closer windows
        target_sim = torch.zeros_like(similarity)
        for i, window in enumerate(windows):
            mask = in_window[:, i]
            target_sim[mask] = 1.0 - (i / len(windows))  # Decreasing target

        # MSE loss between actual and target similarity
        loss = F.mse_loss(similarity, target_sim)

        return loss

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        time_diff_pos: torch.Tensor,
        time_diff_neg: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss components for logging
        """
        # Compute individual losses
        semantic_loss = self.semantic_contrastive_loss(anchor, positive, negative)
        temporal_loss = self.temporal_proximity_loss(
            anchor, positive, negative, time_diff_pos, time_diff_neg
        )
        window_loss = self.time_window_membership_loss(anchor, positive, time_diff_pos)

        # Combined loss
        total_loss = (
            semantic_loss +
            self.temporal_weight * temporal_loss +
            self.window_weight * window_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'semantic': semantic_loss.item(),
            'temporal': temporal_loss.item(),
            'window': window_loss.item()
        }

        return total_loss, loss_dict


def train_epoch(
    model: TemporalEncoder,
    dataloader: DataLoader,
    criterion: TemporalLosses,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_losses = {'total': 0, 'semantic': 0, 'temporal': 0, 'window': 0}

    for batch in tqdm(dataloader, desc="Training"):
        # Move to device
        anchor_text = batch['anchor_text']
        positive_text = batch['positive_text']
        negative_text = batch['negative_text']

        anchor_time = batch['anchor_time'].to(device)
        positive_time = batch['positive_time'].to(device)
        negative_time = batch['negative_time'].to(device)

        time_diff_pos = batch['time_diff_pos'].to(device)
        time_diff_neg = batch['time_diff_neg'].to(device)

        # Forward pass
        anchor_emb = model.encode(anchor_text, anchor_time)
        positive_emb = model.encode(positive_text, positive_time)
        negative_emb = model.encode(negative_text, negative_time)

        # Compute loss
        loss, loss_dict = criterion(
            anchor_emb, positive_emb, negative_emb,
            time_diff_pos, time_diff_neg
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses
        for k, v in loss_dict.items():
            total_losses[k] += v

    # Average losses
    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in total_losses.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--base_model', type=str, default='Alibaba-NLP/gte-base-en-v1.5')
    parser.add_argument('--time_encoding', type=str, default='sinusoidal', choices=['sinusoidal', 'bucketed'])
    parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'film', 'addition'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='temporal_encoding/checkpoints')
    args = parser.parse_args()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Dataset
    dataset = TemporalConversationDataset(args.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Model
    model = TemporalEncoder(
        base_model=args.base_model,
        time_encoding=args.time_encoding,
        fusion_method=args.fusion_method,
        freeze_base=False  # Fine-tune the base encoder
    ).to(device)

    # Loss and optimizer
    criterion = TemporalLosses()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        losses = train_epoch(model, dataloader, criterion, optimizer, device)

        print(f"Losses: {losses}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses
        }, f"{args.output_dir}/checkpoint_epoch_{epoch}.pt")

    print("Training complete!")


if __name__ == '__main__':
    main()
