import os
import sys
import argparse
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def load_summaries_from_folder(folder_path):
    """Load all text files from a folder and return as a dictionary."""
    summaries = {}
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return summaries

    for fname in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, fname)
        if os.path.isfile(file_path) and fname.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    summaries[fname] = f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")

    return summaries


def evaluate_summaries(generated_dir, reference_dir, detailed=False):
    """Evaluate generated summaries against reference summaries."""

    # Load summaries from both directories
    generated_summaries = load_summaries_from_folder(generated_dir)
    reference_summaries = load_summaries_from_folder(reference_dir)

    if not generated_summaries:
        print(f"Error: No summaries found in {generated_dir}")
        return

    if not reference_summaries:
        print(f"Error: No reference summaries found in {reference_dir}")
        return

    # Find common files
    common_files = set(generated_summaries.keys()) & set(
        reference_summaries.keys())

    if not common_files:
        print("Error: No matching files found between the two directories")
        print(f"Generated files: {list(generated_summaries.keys())}")
        print(f"Reference files: {list(reference_summaries.keys())}")
        return

    print(f"Evaluating {len(common_files)} pairs of summaries...")

    # Prepare lists for evaluation
    generated_texts = []
    reference_texts = []

    for fname in sorted(common_files):
        generated_texts.append(generated_summaries[fname])
        reference_texts.append(reference_summaries[fname])
        print(f"  - {fname}")

    # Evaluate with ROUGE
    print("\n" + "="*50)
    print("ROUGE SCORES")
    print("="*50)

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougel_scores = [], [], []

    for i, (gen, ref) in enumerate(zip(generated_texts, reference_texts)):
        scores = scorer.score(ref, gen)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougel_scores.append(scores['rougeL'].fmeasure)

    print(f"ROUGE-1: {sum(rouge1_scores)/len(rouge1_scores):.4f}")
    print(f"ROUGE-2: {sum(rouge2_scores)/len(rouge2_scores):.4f}")
    print(f"ROUGE-L: {sum(rougel_scores)/len(rougel_scores):.4f}")

    # Evaluate with BERTScore
    print("\n" + "="*50)
    print("BERTSCORE")
    print("="*50)

    try:
        P, R, F1 = bert_score(reference_texts, generated_texts,
                              lang="vi", rescale_with_baseline=True)
        print(f"BERTScore Precision: {P.mean().item():.4f}")
        print(f"BERTScore Recall: {R.mean().item():.4f}")
        print(f"BERTScore F1: {F1.mean().item():.4f}")
    except Exception as e:
        print(f"Error computing BERTScore: {e}")

    if detailed:
        # Individual file scores (optional detailed output)
        print("\n" + "="*50)
        print("DETAILED SCORES BY FILE")
        print("="*50)

        for i, fname in enumerate(sorted(common_files)):
            print(f"\n{fname}:")
            print(f"  ROUGE-1: {rouge1_scores[i]:.4f}")
            print(f"  ROUGE-2: {rouge2_scores[i]:.4f}")
            print(f"  ROUGE-L: {rougel_scores[i]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated summaries against reference summaries using ROUGE and BERTScore metrics."
    )
    parser.add_argument(
        "generated_dir",
        help="Directory containing generated summaries"
    )
    parser.add_argument(
        "reference_dir",
        help="Directory containing reference summaries"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed scores for each file"
    )

    args = parser.parse_args()

    if not os.path.exists(args.generated_dir):
        print(
            f"Error: Generated summaries directory '{args.generated_dir}' does not exist")
        sys.exit(1)

    if not os.path.exists(args.reference_dir):
        print(
            f"Error: Reference summaries directory '{args.reference_dir}' does not exist")
        sys.exit(1)

    print(f"Generated summaries directory: {args.generated_dir}")
    print(f"Reference summaries directory: {args.reference_dir}")
    print()

    evaluate_summaries(args.generated_dir, args.reference_dir, args.detailed)


if __name__ == "__main__":
    main()
