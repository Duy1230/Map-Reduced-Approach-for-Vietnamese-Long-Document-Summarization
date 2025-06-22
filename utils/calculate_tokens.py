import os
import json
import argparse
from transformers import AutoTokenizer


def count_stats(text, tokenizer):
    """Count tokens, characters, and words for a given text."""
    # Count tokens
    tokens = tokenizer.encode(text)
    token_count = len(tokens)

    # Count characters (including spaces)
    char_count = len(text)

    # Count words (split by whitespace)
    word_count = len(text.split())

    return token_count, char_count, word_count


def process_folder(folder_path, tokenizer_name="Qwen/Qwen3-4B"):
    """Process all txt files in a folder and return statistics."""
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    results = []

    # Get all txt files
    txt_files = [f for f in os.listdir(
        folder_path) if f.lower().endswith('.txt')]

    print(f"Found {len(txt_files)} txt files to process")

    for filename in sorted(txt_files):
        file_path = os.path.join(folder_path, filename)

        print(f"Processing: {filename}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            token_count, char_count, word_count = count_stats(text, tokenizer)

            file_stats = {
                "filename": filename,
                "path": file_path,
                "tokens": token_count,
                "characters": char_count,
                "words": word_count
            }

            results.append(file_stats)

            print(
                f"  Tokens: {token_count:,}, Characters: {char_count:,}, Words: {word_count:,}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate tokens, characters, and words for txt files")
    parser.add_argument("--folder", required=True,
                        help="Folder containing txt files")
    parser.add_argument("--output", default="file_stats.json",
                        help="Output JSON file")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B",
                        help="Tokenizer model to use")

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist")
        return

    # Process all files
    results = process_folder(args.folder, args.tokenizer)

    # Calculate summary statistics
    total_files = len(results)
    total_tokens = sum(r["tokens"] for r in results)
    total_chars = sum(r["characters"] for r in results)
    total_words = sum(r["words"] for r in results)

    # Create final output
    output_data = {
        "summary": {
            "total_files": total_files,
            "total_tokens": total_tokens,
            "total_characters": total_chars,
            "total_words": total_words,
            "average_tokens_per_file": total_tokens / total_files if total_files > 0 else 0,
            "average_characters_per_file": total_chars / total_files if total_files > 0 else 0,
            "average_words_per_file": total_words / total_files if total_files > 0 else 0
        },
        "files": results
    }

    # Save to JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSummary:")
    print(f"Total files: {total_files}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total words: {total_words:,}")
    print(
        f"Average tokens per file: {total_tokens / total_files:.1f}" if total_files > 0 else "No files processed")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
