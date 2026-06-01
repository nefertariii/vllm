#!/usr/bin/env python3
"""
Prepare liyucheng/arxiv-march-2023 for vLLM benchmark_serving.

Mirrors the request-generation logic in Jenga's benchmark_serving.py
(see https://github.com/heheda12345/Jenga-SOSP25-AE) so that experiments
are directly comparable.

Prompts are: article_text + question  (long shared prefix per article).
Each article contributes `--num-questions` requests, creating a natural
prefix-sharing workload for prefix-aware scheduling evaluation.

Two pre-defined article-ID lists, taken verbatim from Jenga:
  gemma2    – 151 individual articles (fits Gemma-2-9b-it's 8192-token window)
  ministral – 100 groups of concatenated articles (for 32k+ context models
              such as Llama-3.1-8B-Instruct or Ministral-8B)

Output is ShareGPT JSON that can be fed to vLLM's benchmark pipeline:
    python -m vllm.entrypoints.cli.main bench serve \\
        --dataset-name sharegpt --dataset-path <output> \\
        --sharegpt-output-len <fixed_output_len> ...

Or use Jenga's benchmark_serving.py directly with --dataset-name hf.

Usage:
    # Gemma-2-9b-it (single articles)
    python scripts/prepare_arxiv_qa.py \\
        --model-type gemma2 \\
        --num-prompts 80 \\
        --fixed-output-len 150 \\
        --output benchmark_data/arxiv_gemma2.json

    # Llama-3.1-8B-Instruct (grouped / concatenated articles)
    python scripts/prepare_arxiv_qa.py \\
        --model-type llama \\
        --num-prompts 32 \\
        --fixed-output-len 150 \\
        --max-len 80000 \\
        --output benchmark_data/arxiv_llama.json
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# Article-ID lists taken verbatim from Jenga's benchmark_serving.py
# ---------------------------------------------------------------------------

def _gemma2_article_ids() -> List[int]:
    """151 individual article indices for Gemma-2-style prompts."""
    return [
        0, 1, 2, 3, 7, 10, 21, 25, 39, 43, 46, 51, 52, 53, 54, 58, 59, 65, 67,
        70, 71, 74, 79, 80, 82, 85, 96, 97, 101, 103, 108, 109, 111, 121, 122,
        124, 127, 128, 130, 131, 132, 133, 138, 140, 141, 143, 147, 151, 152,
        156, 157, 163, 164, 168, 169, 171, 173, 179, 180, 182, 184, 186, 188,
        189, 192, 193, 194, 198, 205, 206, 209, 210, 215, 220, 222, 223, 231,
        233, 236, 245, 247, 255, 261, 264, 272, 274, 279, 280, 283, 285, 286,
        287, 296, 298, 299, 301, 302, 306, 308, 314, 315, 318, 319, 324, 329,
        331, 342, 343, 346, 347, 351, 356, 357, 360, 362, 366, 369, 375, 379,
        388, 389, 390, 393, 395, 396, 397, 400, 401, 402, 404, 416, 417, 421,
        426, 428, 429, 430, 433, 436, 439, 440, 443, 448, 453, 456, 457, 464,
        466, 467, 469, 470, 473, 474, 475, 476, 477, 479, 484, 486, 487,
    ]


def _ministral_article_ids() -> List[List[int]]:
    """100 groups of article indices, each group is concatenated for longer
    prompts suitable for Ministral-8B / Llama-3.1-8B-Instruct."""
    return [
        [0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13],
        [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25],
        [26, 27, 28, 29], [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
        [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68],
        [69, 70, 71, 72, 73, 74, 75, 76, 77],
        [78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88], [89, 90, 91, 92, 93],
        [94, 95, 96, 97, 98, 99, 100, 101],
        [102, 103, 104, 105, 106, 107, 108, 109], [110, 111, 112],
        [113, 114], [115, 116, 117], [118, 119, 120, 121, 122],
        [123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135],
        [136, 137, 138, 139, 140, 141, 142, 143, 144, 145],
        [146, 147, 148, 149, 150, 151, 152, 153, 154],
        [155, 156, 157, 158, 159, 160, 161, 162, 163, 164],
        [165, 166, 167, 168, 169, 170, 171, 172, 173, 174],
        [175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185],
        [186, 187, 188, 189, 190, 191, 192, 193, 194, 195],
        [196, 197, 198, 199, 200, 201, 202, 203, 204, 205],
        [206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216],
        [217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227],
        [228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238],
        [239, 240, 241], [242, 243, 244, 245, 246],
        [247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257],
        [258, 259, 260, 261, 262, 263, 264, 265],
        [266, 267, 268, 269, 270, 271, 272, 273, 274, 275],
        [276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286],
        [287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298],
        [299, 300, 301, 302, 303, 304, 305, 306],
        [307, 308, 309, 310, 311, 312, 313, 314, 315],
        [316, 317, 318, 319, 320, 321],
        [322, 323, 324, 325, 326, 327, 328, 329, 330, 331],
        [332, 333, 334], [335, 336, 337, 338, 339],
        [340, 341, 342, 343, 344], [345, 346, 347, 348, 349],
        [350, 351, 352, 353, 354, 355],
        [356, 357, 358, 359, 360, 361, 362, 363, 364],
        [365, 366, 367, 368, 369], [370, 371, 372, 373, 374, 375],
        [376, 377, 378, 379],
        [380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391],
        [392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404],
        [405, 406, 407, 408, 409, 410], [411, 412, 413, 414, 415, 416],
        [417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428],
        [429, 430, 431, 432, 433, 434, 435, 436],
        [437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448],
        [449, 450, 451, 452, 453, 454, 455],
        [456, 457, 458, 459, 460, 461, 462, 463],
        [464, 465, 466, 467, 468, 469, 470],
        [471, 472, 473, 474, 475, 476, 477, 478, 479, 480],
        [481, 482, 483, 484, 485, 486, 487, 488],
        [489, 490, 491, 492, 493, 494, 495, 496, 497, 498],
    ]


# Questions used per article, matching Jenga's `questions` list.
QUESTIONS = [
    'What is the author of the paper?',
    'What is the title of the paper?',
    'What is the abstract of the paper?',
    'What is the introduction of the paper?',
    'Please summarize the paper.',
    'What is the conclusion of the paper?',
    'What is the main contribution of the paper?',
    'When was the paper published?',
    'What is the categories of the paper?',
    'What is the URL of the paper?',
]


def _build_article_text(
    dataset,
    article_id: Union[int, List[int]],
    max_len_chars: Optional[int] = None,
) -> str:
    """Return the article text for a single index or concatenated group."""
    if isinstance(article_id, list):
        text = ""
        for idx in article_id:
            candidate = text + dataset[idx]['text']
            if max_len_chars is not None and len(candidate) > max_len_chars:
                break
            text = candidate
        return text
    return dataset[article_id]['text']


def generate_requests(
    dataset,
    article_ids: Union[List[int], List[List[int]]],
    num_questions_per_article: int,
    shuffle_range: int,
    seed: int,
    max_len_chars: Optional[int] = None,
) -> List[dict]:
    """
    Generate ShareGPT-format conversations from the arxiv dataset.

    Each article contributes `num_questions_per_article` prompts.  Prompts
    within each batch of `shuffle_range` articles are shuffled together so
    same-article requests are interleaved — creating the workload where a
    prefix-aware scheduler must discover the shared prefix.

    Returns a list of ShareGPT conversation dicts.
    """
    rng = random.Random(seed)
    conversations: List[dict] = []
    requests_local: List[str] = []
    j = 0

    target = shuffle_range * num_questions_per_article
    while len(requests_local) < target:
        if j >= len(article_ids):
            break
        article_id = article_ids[j]
        article_text = _build_article_text(dataset, article_id, max_len_chars)

        batch: List[str] = []
        too_long = False
        for q in QUESTIONS[:num_questions_per_article]:
            prompt = article_text + q
            if max_len_chars is not None and len(prompt) > max_len_chars:
                too_long = True
                break
            batch.append(prompt)

        if too_long:
            j += 1
            continue

        requests_local.extend(batch)
        j += 1

    rng.shuffle(requests_local)

    for prompt in requests_local:
        conversations.append({
            "conversations": [
                {"from": "human", "value": prompt},
                # Placeholder — actual output length is controlled by
                # --sharegpt-output-len at benchmark time.
                {"from": "gpt", "value": ""},
            ]
        })

    return conversations


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-type",
        choices=["gemma2", "llama"],
        default="gemma2",
        help=(
            "gemma2: single articles, fits 8192-token context "
            "(Gemma-2-9b-it).  "
            "llama: grouped/concatenated articles, fits 32k+ context "
            "(Llama-3.1-8B-Instruct, Ministral-8B)."
        ),
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=80,
        help="Total number of prompts to generate (must be divisible by "
             "--num-questions).",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=4,
        help="Questions per article (default 4, max 10).",
    )
    parser.add_argument(
        "--fixed-output-len",
        type=int,
        default=150,
        help="Fixed output length placeholder (tokens).  Pass the same "
             "value as --sharegpt-output-len when running the benchmark.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Approximate max prompt length in characters.  Used to skip "
             "articles that would exceed the model context window.  "
             "Jenga uses 80000 chars for ministral.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_data/arxiv_qa.json",
        help="Output path for the ShareGPT-format JSON.",
    )
    parser.add_argument("--seed", type=int, default=55555)
    args = parser.parse_args()

    assert args.num_questions <= len(QUESTIONS), (
        f"--num-questions must be <= {len(QUESTIONS)}"
    )
    assert args.num_prompts % args.num_questions == 0, (
        "--num-prompts must be divisible by --num-questions"
    )

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("ERROR: install 'datasets':  pip install datasets")

    print("Loading liyucheng/arxiv-march-2023 (train split)...", flush=True)
    dataset = load_dataset("liyucheng/arxiv-march-2023", name="", split="train")
    print(f"Loaded {len(dataset)} articles.", flush=True)

    if args.model_type == "gemma2":
        article_ids: Union[List[int], List[List[int]]] = _gemma2_article_ids()
    else:
        article_ids = _ministral_article_ids()

    shuffle_range = args.num_prompts // args.num_questions
    conversations = generate_requests(
        dataset=dataset,
        article_ids=article_ids,
        num_questions_per_article=args.num_questions,
        shuffle_range=shuffle_range,
        seed=args.seed,
        max_len_chars=args.max_len,
    )

    if len(conversations) < args.num_prompts:
        print(
            f"WARNING: only generated {len(conversations)} prompts "
            f"(requested {args.num_prompts}).  "
            "Try reducing --num-prompts or --num-questions."
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

    prompt_lens = [len(c["conversations"][0]["value"]) for c in conversations]
    print(f"\nDataset statistics:")
    print(f"  Total conversations   : {len(conversations)}")
    print(f"  Article-ID list       : {args.model_type}")
    print(f"  Questions per article : {args.num_questions}")
    print(f"  Fixed output len      : {args.fixed_output_len} tokens")
    if prompt_lens:
        print(f"  Avg prompt length     : {sum(prompt_lens)/len(prompt_lens):,.0f} chars")
        print(f"  Min/Max prompt length : {min(prompt_lens):,} / {max(prompt_lens):,} chars")
    print(f"\nSaved to: {args.output}")
    print(
        f"\nRun the benchmark with (example):\n"
        f"  python /home/r14922144/Jenga-SOSP25-AE/benchmark_serving.py \\\n"
        f"      --port 8000 \\\n"
        f"      --model <model-id> \\\n"
        f"      --dataset-path liyucheng/arxiv-march-2023 \\\n"
        f"      --dataset-name hf \\\n"
        f"      --hf-subset {'gemma2' if args.model_type == 'gemma2' else 'ministral'} \\\n"
        f"      --hf-split train \\\n"
        f"      --num_prompts {args.num_prompts} \\\n"
        f"      --seed {args.seed} \\\n"
        f"      --hf-output-len {args.fixed_output_len} \\\n"
        f"      {'--hf-max-len ' + str(args.max_len) + ' \\' if args.max_len else ''}\n"
        f"      --ignore-eos"
    )


if __name__ == "__main__":
    main()
