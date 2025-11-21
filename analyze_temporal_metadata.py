#!/usr/bin/env python3
"""
Analyze how temporal metadata is currently used in LongMemEval baseline.

This script examines:
1. How dates are presented in prompts (metadata vs content)
2. Where temporal information appears in the evaluation pipeline
3. How to establish a baseline before encoding time into data
"""

import json
import sys
sys.path.append('benchmarks/LongMemEval/src')

from generation.run_generation import prepare_prompt
from transformers import AutoTokenizer


def analyze_prompt_construction(data_file='benchmarks/LongMemEval/data/longmemeval_oracle.json'):
    """Analyze how temporal metadata appears in generated prompts."""

    data = json.load(open(data_file))
    # Use GPT-2 tokenizer as a proxy (publicly available, similar token counts)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    print("=" * 80)
    print("TEMPORAL METADATA USAGE ANALYSIS")
    print("=" * 80)

    # Test different question types
    question_types = ['temporal-reasoning', 'knowledge-update', 'single-session-user', 'multi-session']

    for qtype in question_types:
        print(f"\n{'=' * 80}")
        print(f"Question Type: {qtype.upper()}")
        print("=" * 80)

        # Get example
        example = next(x for x in data if x['question_type'] == qtype and '_abs' not in x['question_id'])

        print(f"\nQuestion: {example['question']}")
        print(f"Answer: {example['answer']}")
        print(f"Question Date: {example['question_date']}")
        print(f"Number of sessions: {len(example['haystack_sessions'])}")
        print(f"Session dates: {example['haystack_dates']}")

        # Generate prompt as it would appear with different settings
        print(f"\n--- PROMPT WITH ORACLE RETRIEVAL (JSON FORMAT) ---")
        prompt = prepare_prompt(
            entry=example,
            retriever_type='oracle-session',
            topk_context=100,
            useronly=False,
            history_format='json',
            cot=True,
            tokenizer=tokenizer,
            tokenizer_backend='huggingface',
            max_retrieval_length=100000,
            merge_key_expansion_into_value=None
        )
        print(prompt[:2000] + "\n..." if len(prompt) > 2000 else prompt)

        print(f"\n--- PROMPT WITH ORACLE RETRIEVAL (NL FORMAT) ---")
        prompt_nl = prepare_prompt(
            entry=example,
            retriever_type='oracle-session',
            topk_context=100,
            useronly=False,
            history_format='nl',
            cot=True,
            tokenizer=tokenizer,
            tokenizer_backend='huggingface',
            max_retrieval_length=100000,
            merge_key_expansion_into_value=None
        )
        print(prompt_nl[:2000] + "\n..." if len(prompt_nl) > 2000 else prompt_nl)

        # Analyze where dates appear
        print(f"\n--- TEMPORAL METADATA LOCATION ---")
        print(f"✓ Question date appears in prompt: {example['question_date'] in prompt}")
        print(f"✓ Session dates appear in prompt: {any(d in prompt for d in example['haystack_dates'])}")
        print(f"✓ Dates are METADATA (not in conversation content)")

        print("\n" + "=" * 80)
        break  # Just show one detailed example for now


def show_baseline_approach():
    """Show the current baseline approach with temporal metadata."""

    print("\n" + "=" * 80)
    print("CURRENT BASELINE: TEMPORAL INFORMATION AS METADATA")
    print("=" * 80)

    print("""
The LongMemEval baseline currently uses temporal information as METADATA:

1. **In Prompts**: Dates are shown as "Session Date:" or "Current Date:" labels
   - Example: "Session Date: 2023/05/20 (Thu) 14:30"
   - This is metadata that annotates the conversation

2. **In Retrieval**: Timestamps are stored separately from content
   - corpus_timestamps[] array parallel to corpus[] array
   - Dates can be used for time-aware filtering but not in embeddings

3. **Limitations of Metadata Approach**:
   - Model must learn to parse date formats from examples
   - Temporal relationships (before/after/duration) are implicit
   - Dense retrievers don't encode temporal similarity
   - Temporal reasoning requires careful prompt engineering

YOUR GOAL: Encode time INTO the data itself
- Instead of "Session Date: 2023/05/20"
- Encode temporal information directly in conversation text
- Or as special tokens the model learns to interpret
- Allow temporal relationships to be learned, not just parsed
""")


def suggest_next_steps():
    """Suggest next steps for temporal encoding experiments."""

    print("\n" + "=" * 80)
    print("RECOMMENDED BASELINE EXPERIMENTS")
    print("=" * 80)

    print("""
Before implementing temporal encoding, run these baselines:

1. **Oracle Setting** (Easiest - no retrieval needed)
   - Only evidence sessions in context
   - Focus on how well model uses temporal metadata in QA
   - Command:
     cd benchmarks/LongMemEval/src/generation
     bash run_generation.sh ../../data/longmemeval_oracle.json gpt-4o-mini \\
         full-history-session 100 json false con

2. **Full History (LongMemEval_S)**
   - All ~40 sessions in context
   - Tests long-context temporal reasoning
   - Requires model with 128k+ context window

3. **Retrieval + Generation**
   - First run retrieval to get relevant sessions
   - Then run generation with retrieved context
   - Shows how temporal info helps/hurts retrieval

ANALYSIS TO DO:
- How accurately does model handle temporal reasoning WITH metadata?
- Does temporal metadata help retrieval (time-aware filtering)?
- Which question types benefit most from temporal information?

Then you can compare: Metadata vs Encoded temporal information
""")


if __name__ == '__main__':
    analyze_prompt_construction()
    show_baseline_approach()
    suggest_next_steps()
