#!/usr/bin/env python3
"""
Script to summarize documents using Qwen3-8B and evaluate summarization quality.
Requirements:
    pip install transformers torch rouge_score bert-score
Usage:
    python run_summarization.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def generate_summary(model, tokenizer, text, max_input_tokens, max_new_tokens, device):
    prompt = f"Please summary the following document.\n\n{text}"
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Apply chat template with thinking disabled
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Disable thinking mode for Qwen3
    )

    # Tokenize and truncate the formatted text
    inputs = tokenizer(
        formatted_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    ).to(device)

    # Generate summary
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    # Decode generated summary (skip input tokens)
    summary = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return summary


def main():
    model_name = "Qwen/Qwen3-8B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # Context settings
    max_context = 32768
    max_new_tokens = 1024
    max_input_tokens = max_context - max_new_tokens

    docs_dir = "data/doc"
    summary_dir = "data/summary"
    generated_summaries = []
    references = []

    # Iterate over documents
    for fname in sorted(os.listdir(docs_dir)):
        doc_path = os.path.join(docs_dir, fname)
        ref_path = os.path.join(summary_dir, fname)
        if not os.path.isfile(ref_path):
            print(f"No reference summary for {fname}, skipping")
            continue
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_text = f.read()
        with open(ref_path, 'r', encoding='utf-8') as f:
            ref = f.read()

        # Truncate document text to fit within max_input_tokens
        doc_tokens = tokenizer.encode(doc_text)
        if len(doc_tokens) > max_input_tokens:
            # Keep only the first max_input_tokens and decode back to text
            truncated_tokens = doc_tokens[:max_input_tokens]
            doc_text = tokenizer.decode(
                truncated_tokens, skip_special_tokens=True)
            print(
                f"Processing {fname}... (truncated from {len(doc_tokens)} to {len(truncated_tokens)} tokens)")
        else:
            print(f"Processing {fname}... ({len(doc_tokens)} tokens)")

        # Generate and collect summaries
        summary = generate_summary(
            model, tokenizer, doc_text, max_input_tokens, max_new_tokens, device)
        generated_summaries.append(summary)
        references.append(ref)

    # Evaluate with ROUGE
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for gen, ref in zip(generated_summaries, references):
        scores = scorer.score(ref, gen)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    print("ROUGE-1: {:.4f}".format(sum(rouge1_scores)/len(rouge1_scores)))
    print("ROUGE-2: {:.4f}".format(sum(rouge2_scores)/len(rouge2_scores)))
    print("ROUGE-L: {:.4f}".format(sum(rougeL_scores)/len(rougeL_scores)))

    # Evaluate with BERTScore
    P, R, F1 = bert_score(references, generated_summaries,
                          lang="en", rescale_with_baseline=True)
    print("BERTScore P: {:.4f}, R: {:.4f}, F1: {:.4f}".format(
        P.mean().item(), R.mean().item(), F1.mean().item()))


if __name__ == "__main__":
    main()
