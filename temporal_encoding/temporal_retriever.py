"""
Temporal Retrieval Encoder for LongMemEval

This module implements temporal-aware dense retrieval by encoding both
content (conversation text) and time (timestamps) into a unified embedding space.

Usage:
    # Replace LongMemEval's flat-gte/flat-stella with temporal-gte
    encoder = TemporalEncoder(base_model='Alibaba-NLP/gte-base-en-v1.5')
    embeddings = encoder.encode(texts, timestamps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import List, Union, Optional


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embeddings similar to positional encoding in Transformers.

    Encodes continuous timestamps using sine/cosine functions at different frequencies.
    This allows the model to learn temporal patterns (daily/weekly/seasonal cycles).
    """

    def __init__(self, embedding_dim: int = 128, max_period: float = 10000.0):
        """
        Args:
            embedding_dim: Dimension of time embedding (must be even)
            max_period: Maximum period for sinusoidal functions
        """
        super().__init__()
        assert embedding_dim % 2 == 0, "embedding_dim must be even"

        self.embedding_dim = embedding_dim
        self.max_period = max_period

        # Precompute frequency bands
        half_dim = embedding_dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: Unix timestamps [batch_size] or relative time in seconds

        Returns:
            Time embeddings [batch_size, embedding_dim]
        """
        # timestamps shape: [batch_size]
        # freqs shape: [embedding_dim // 2]

        # Compute angles
        angles = timestamps.unsqueeze(-1) * self.freqs.unsqueeze(0)  # [batch, dim//2]

        # Apply sin and cos
        sin_emb = torch.sin(angles)
        cos_emb = torch.cos(angles)

        # Concatenate
        time_emb = torch.cat([sin_emb, cos_emb], dim=-1)  # [batch, dim]

        return time_emb


class BucketedTimeEmbedding(nn.Module):
    """
    Bucketed time embeddings for relative temporal distances.

    Maps time differences into log-spaced buckets:
    - Recent: < 1 day
    - This week: 1-7 days
    - This month: 7-30 days
    - Months ago: 30-365 days
    - Years ago: > 365 days
    """

    def __init__(self, embedding_dim: int = 128, num_buckets: int = 32):
        """
        Args:
            embedding_dim: Dimension of time embedding
            num_buckets: Number of temporal buckets
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, embedding_dim)

    def get_bucket(self, time_diff_seconds: torch.Tensor) -> torch.Tensor:
        """
        Convert time difference to bucket index using log spacing.

        Args:
            time_diff_seconds: Time difference in seconds [batch_size]

        Returns:
            Bucket indices [batch_size]
        """
        # Convert to days
        time_diff_days = time_diff_seconds / 86400.0

        # Log-space bucketing
        # Bucket 0: same day (0-1 days)
        # Bucket 1-10: 1-365 days (log-spaced)
        # Bucket 11+: > 1 year (log-spaced)

        # Clamp to avoid log(0)
        time_diff_days = torch.clamp(time_diff_days, min=1e-5)

        # Log-scale bucket assignment
        log_days = torch.log(time_diff_days + 1)
        buckets = (log_days * self.num_buckets / 10.0).long()
        buckets = torch.clamp(buckets, 0, self.num_buckets - 1)

        return buckets

    def forward(self, time_diff_seconds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_diff_seconds: Time differences in seconds [batch_size]

        Returns:
            Time embeddings [batch_size, embedding_dim]
        """
        buckets = self.get_bucket(time_diff_seconds)
        return self.embedding(buckets)


class FusionModule(nn.Module):
    """
    Fusion module to combine content and temporal embeddings.

    Supports multiple fusion strategies:
    - concat: Concatenate and project
    - film: Feature-wise Linear Modulation
    - addition: Simple element-wise addition (if dims match)
    """

    def __init__(
        self,
        content_dim: int,
        time_dim: int,
        output_dim: int,
        fusion_method: str = 'concat'
    ):
        """
        Args:
            content_dim: Dimension of content embeddings
            time_dim: Dimension of time embeddings
            output_dim: Final output dimension
            fusion_method: 'concat', 'film', or 'addition'
        """
        super().__init__()
        self.fusion_method = fusion_method

        if fusion_method == 'concat':
            self.projection = nn.Sequential(
                nn.Linear(content_dim + time_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
        elif fusion_method == 'film':
            # FiLM: Feature-wise Linear Modulation
            self.gamma_net = nn.Linear(time_dim, content_dim)
            self.beta_net = nn.Linear(time_dim, content_dim)
            if content_dim != output_dim:
                self.projection = nn.Linear(content_dim, output_dim)
            else:
                self.projection = nn.Identity()
        elif fusion_method == 'addition':
            assert time_dim == content_dim, "Addition requires matching dimensions"
            if content_dim != output_dim:
                self.projection = nn.Linear(content_dim, output_dim)
            else:
                self.projection = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    def forward(self, content_emb: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content_emb: Content embeddings [batch, content_dim]
            time_emb: Time embeddings [batch, time_dim]

        Returns:
            Fused embeddings [batch, output_dim]
        """
        if self.fusion_method == 'concat':
            combined = torch.cat([content_emb, time_emb], dim=-1)
            return self.projection(combined)

        elif self.fusion_method == 'film':
            # FiLM: z' = gamma(t) * z + beta(t)
            gamma = self.gamma_net(time_emb)
            beta = self.beta_net(time_emb)
            modulated = gamma * content_emb + beta
            return self.projection(modulated)

        elif self.fusion_method == 'addition':
            combined = content_emb + time_emb
            return self.projection(combined)


class TemporalEncoder(nn.Module):
    """
    Main temporal retrieval encoder.

    Combines a pre-trained sentence encoder (GTE/E5) with temporal encodings
    to create time-aware embeddings for conversational memory retrieval.

    Example:
        encoder = TemporalEncoder('Alibaba-NLP/gte-base-en-v1.5')
        embeddings = encoder.encode(
            texts=["I got my car serviced", "GPS not working"],
            timestamps=[1678886400, 1678972800]  # Unix timestamps
        )
    """

    def __init__(
        self,
        base_model: str = 'Alibaba-NLP/gte-base-en-v1.5',
        time_encoding: str = 'sinusoidal',  # 'sinusoidal' or 'bucketed'
        time_dim: int = 128,
        fusion_method: str = 'concat',
        freeze_base: bool = False
    ):
        """
        Args:
            base_model: HuggingFace model name for content encoding
            time_encoding: Type of temporal encoding ('sinusoidal' or 'bucketed')
            time_dim: Dimension of time embeddings
            fusion_method: How to combine content and time ('concat', 'film', 'addition')
            freeze_base: Whether to freeze the base content encoder
        """
        super().__init__()

        # Content encoder (GTE, E5, etc.)
        self.content_encoder = SentenceTransformer(base_model)
        self.content_dim = self.content_encoder.get_sentence_embedding_dimension()

        if freeze_base:
            for param in self.content_encoder.parameters():
                param.requires_grad = False

        # Time encoder
        if time_encoding == 'sinusoidal':
            self.time_encoder = SinusoidalTimeEmbedding(time_dim)
        elif time_encoding == 'bucketed':
            self.time_encoder = BucketedTimeEmbedding(time_dim)
        else:
            raise ValueError(f"Unknown time encoding: {time_encoding}")

        # Fusion module
        self.fusion = FusionModule(
            content_dim=self.content_dim,
            time_dim=time_dim,
            output_dim=self.content_dim,  # Keep same dim for compatibility
            fusion_method=fusion_method
        )

    def encode(
        self,
        texts: Union[str, List[str]],
        timestamps: Union[float, List[float], torch.Tensor],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode texts with temporal information.

        Args:
            texts: Single text or list of texts
            timestamps: Unix timestamps or relative time in seconds
            normalize: Whether to L2-normalize output embeddings

        Returns:
            Temporal-aware embeddings [batch, content_dim]
        """
        # Handle single inputs
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(timestamps, (int, float)):
            timestamps = [timestamps]

        # Convert timestamps to tensor
        if not isinstance(timestamps, torch.Tensor):
            timestamps = torch.tensor(timestamps, dtype=torch.float32)

        # Encode content
        content_emb = self.content_encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # Encode time
        time_emb = self.time_encoder(timestamps.to(content_emb.device))

        # Fuse
        embeddings = self.fusion(content_emb, time_emb)

        # Normalize if requested
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def forward(self, texts: List[str], timestamps: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        return self.encode(texts, timestamps, normalize=True)


def timestamp_to_seconds(timestamp_str: str) -> float:
    """
    Convert LongMemEval timestamp to Unix seconds.

    Args:
        timestamp_str: "2023/04/10 (Mon) 14:47" format

    Returns:
        Unix timestamp in seconds
    """
    # Parse: "2023/04/10 (Mon) 14:47"
    date_part = timestamp_str.split(' (')[0]  # "2023/04/10"
    time_part = timestamp_str.split(') ')[1]  # "14:47"

    datetime_str = f"{date_part} {time_part}"
    dt = datetime.strptime(datetime_str, "%Y/%m/%d %H:%M")

    return dt.timestamp()


# Example usage
if __name__ == '__main__':
    # Initialize encoder
    encoder = TemporalEncoder(
        base_model='sentence-transformers/all-MiniLM-L6-v2',  # Small model for testing
        time_encoding='sinusoidal',
        fusion_method='concat'
    )

    # Example: Encode conversation sessions
    texts = [
        "I got my car serviced for the first time",
        "My GPS system is not functioning correctly"
    ]

    timestamps = [
        timestamp_to_seconds("2023/04/10 (Mon) 14:47"),
        timestamp_to_seconds("2023/04/10 (Mon) 17:50")
    ]

    # Get temporal-aware embeddings
    embeddings = encoder.encode(texts, timestamps)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First embedding (first 5 dims): {embeddings[0, :5]}")

    # Compute similarity
    similarity = F.cosine_similarity(embeddings[0:1], embeddings[1:2])
    print(f"Similarity between sessions: {similarity.item():.4f}")
