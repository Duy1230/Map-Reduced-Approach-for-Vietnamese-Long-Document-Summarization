import os
import requests
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def generate_summary_ollama(text, tokenizer, max_input_tokens, max_new_tokens, ollama_url, model_name):
    # Truncate document at token level
    tokens = tokenizer.encode(text)
    if len(tokens) > max_input_tokens:
        tokens = tokens[:max_input_tokens]
        text = tokenizer.decode(tokens, skip_special_tokens=True)

    # Prepare prompt for Ollama
    prompt = f"""
    Bạn là một chuyên gia tóm tắt nội dung.
    Vui lòng viết một bản tóm tắt chi tiết cho tài liệu sau bằng **tiếng Việt**.
    \n\n{text}.
    \n\nLưu ý: Không sử dụng dấu đầu dòng, hãy viết bằng câu đầy đủ và theo đoạn văn.
    """

    # Call Ollama native API
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_predict": max_new_tokens
        },

    }

    resp = requests.post(f"{ollama_url}/api/generate", json=payload)
    resp.raise_for_status()
    return resp.json()["response"]


def main():
    OLLAMA_URL = "http://localhost:11434"
    MODEL_NAME = "llama3.2:3b"
    # Tokenizer for truncation
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b")

    # Context settings
    max_context = 16384
    max_new_tokens = 2048
    max_input_tokens = max_context - max_new_tokens

    docs_dir = "data/doc"
    summary_dir = "data/summary"
    generated_summaries_dir = "data/generated_summaries"

    # Create generated summaries directory if it doesn't exist
    os.makedirs(generated_summaries_dir, exist_ok=True)

    generated_summaries = []
    references = []

    for fname in sorted(os.listdir(docs_dir)):
        doc_path = os.path.join(docs_dir, fname)
        ref_path = os.path.join(summary_dir, fname)
        gen_path = os.path.join(generated_summaries_dir, fname)

        # Skip if generated summary already exists
        if os.path.isfile(gen_path):
            print(
                f"Generated summary for {fname} already exists, skipping generation")
            # Load existing generated summary for evaluation
            with open(gen_path, 'r', encoding='utf-8') as f:
                generated_summaries.append(f.read())
            if os.path.isfile(ref_path):
                with open(ref_path, 'r', encoding='utf-8') as f:
                    references.append(f.read())
            continue

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

        print(f"Processing {fname}...")
        summary = generate_summary_ollama(
            doc_text, tokenizer, max_input_tokens, max_new_tokens,
            OLLAMA_URL, MODEL_NAME
        )

        # Save generated summary immediately
        with open(gen_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Saved generated summary to {gen_path}")

        generated_summaries.append(summary)
        references.append(ref)

    # Evaluate with ROUGE
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougel = [], [], []
    for gen, ref in zip(generated_summaries, references):
        scores = scorer.score(ref, gen)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougel.append(scores['rougeL'].fmeasure)
    print("ROUGE-1: {:.4f}".format(sum(rouge1)/len(rouge1)))
    print("ROUGE-2: {:.4f}".format(sum(rouge2)/len(rouge2)))
    print("ROUGE-L: {:.4f}".format(sum(rougel)/len(rougel)))

    # Evaluate with BERTScore
    P, R, F1 = bert_score(references, generated_summaries,
                          lang="vi", rescale_with_baseline=True)
    print("BERTScore P: {:.4f}, R: {:.4f}, F1: {:.4f}".format(
        P.mean().item(), R.mean().item(), F1.mean().item()
    ))


if __name__ == "__main__":
    main()
