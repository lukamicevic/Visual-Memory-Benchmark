"""
Integration script: Use Temporal Encoder with LongMemEval Retrieval

This replaces LongMemEval's flat-gte/flat-stella with your temporal-aware encoder.

Usage:
    python integrate_with_longmemeval.py \\
        --checkpoint temporal_encoding/checkpoints/checkpoint_epoch_9.pt \\
        --data benchmarks/LongMemEval/data/longmemeval_s_cleaned.json \\
        --output_dir temporal_retrieval_results
"""

import sys
sys.path.append('benchmarks/LongMemEval/src')

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from temporal_retriever import TemporalEncoder, timestamp_to_seconds


def run_temporal_retrieval(
    encoder: TemporalEncoder,
    query: str,
    query_timestamp: float,
    corpus: list,
    corpus_timestamps: list,
    top_k: int = 10
) -> list:
    """
    Run retrieval using temporal encoder.

    Args:
        encoder: Trained temporal encoder
        query: Query text
        query_timestamp: Query timestamp (Unix seconds)
        corpus: List of corpus texts
        corpus_timestamps: List of corpus timestamps
        top_k: Number of results to return

    Returns:
        List of (index, score) tuples sorted by relevance
    """
    device = next(encoder.parameters()).device

    # Encode query
    query_emb = encoder.encode(query, query_timestamp)  # [1, dim] already batched

    # Encode corpus in batches
    batch_size = 64
    corpus_embs = []

    for i in range(0, len(corpus), batch_size):
        batch_texts = corpus[i:i + batch_size]
        batch_times = corpus_timestamps[i:i + batch_size]

        batch_emb = encoder.encode(batch_texts, batch_times)
        corpus_embs.append(batch_emb)

    corpus_embs = torch.cat(corpus_embs, dim=0)  # [corpus_size, dim]

    # Compute similarities
    similarities = (query_emb @ corpus_embs.T).squeeze(0)  # [corpus_size]

    # Get top-k
    scores, indices = torch.topk(similarities, k=min(top_k, len(corpus)))

    results = [(idx.item(), score.item()) for idx, score in zip(indices, scores)]

    return results


def process_longmemeval_entry(
    entry: dict,
    encoder: TemporalEncoder,
    top_k: int = 50
) -> dict:
    """
    Process one LongMemEval question with temporal retrieval.

    Args:
        entry: LongMemEval question entry
        encoder: Temporal encoder
        top_k: Number of sessions to retrieve

    Returns:
        Entry with retrieval_results added
    """
    # Build corpus
    corpus = []
    corpus_timestamps = []
    corpus_ids = []

    for session, date, sess_id in zip(
        entry['haystack_sessions'],
        entry['haystack_dates'],
        entry['haystack_session_ids']
    ):
        # Convert session to text
        text_parts = []
        for turn in session:
            if turn['role'] == 'user':  # Only index user utterances
                text_parts.append(turn['content'])

        if text_parts:
            corpus.append(' '.join(text_parts))
            corpus_timestamps.append(timestamp_to_seconds(date))
            corpus_ids.append(sess_id)

    # Query
    query = entry['question']
    query_timestamp = timestamp_to_seconds(entry['question_date'])

    # Retrieve
    results = run_temporal_retrieval(
        encoder,
        query,
        query_timestamp,
        corpus,
        corpus_timestamps,
        top_k=top_k
    )

    # Format results
    ranked_items = []
    for idx, score in results:
        ranked_items.append({
            'corpus_id': corpus_ids[idx],
            'text': corpus[idx],
            'timestamp': entry['haystack_dates'][idx],
            'score': score
        })

    # Add to entry
    entry['retrieval_results'] = {
        'query': query,
        'ranked_items': ranked_items,
        'method': 'temporal-encoder'
    }

    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained temporal encoder checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to LongMemEval JSON file')
    parser.add_argument('--base_model', type=str, default='Alibaba-NLP/gte-base-en-v1.5')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Number of sessions to retrieve')
    parser.add_argument('--output_dir', type=str, default='temporal_retrieval_results')
    args = parser.parse_args()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"Loading temporal encoder from {args.checkpoint}")
    encoder = TemporalEncoder(base_model=args.base_model)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder = encoder.to(device)
    encoder.eval()

    print(f"Model loaded (epoch {checkpoint['epoch']})")

    # Load data
    print(f"Loading data from {args.data}")
    data = json.load(open(args.data))
    print(f"Loaded {len(data)} questions")

    # Process each entry
    results = []
    for entry in tqdm(data, desc="Running temporal retrieval"):
        result = process_longmemeval_entry(entry, encoder, top_k=args.top_k)
        results.append(result)

    # Save results
    output_file = f"{args.output_dir}/temporal_retrieval_top{args.top_k}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
    print("\nNow you can run LongMemEval generation with these retrieved sessions:")
    print(f"  cd benchmarks/LongMemEval/src/generation")
    print(f"  bash run_generation.sh {output_file} gpt-4o-mini flat-session {args.top_k}")


if __name__ == '__main__':
    main()
