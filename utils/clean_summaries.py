import os
import sys
import argparse
import re
from pathlib import Path


def clean_thinking_tags(text):
    """Remove <think> tags and their content from text."""
    # Pattern to match <think>...</think> including multiline content
    pattern = r'<think>.*?</think>'

    # Remove the thinking tags and content (case insensitive, multiline, dotall)
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # Clean up extra whitespace and newlines
    # Remove excessive newlines
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()  # Remove leading/trailing whitespace

    return cleaned_text


def process_file(input_path, output_path):
    """Process a single file to remove thinking tags."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if file contains thinking tags
        if '<think>' in content.lower():
            cleaned_content = clean_thinking_tags(content)

            # Write cleaned content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)

            print(f"✓ Cleaned: {input_path.name}")
            return True
        else:
            # If no thinking tags, copy as-is (if different output directory)
            if input_path != output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            print(f"- No changes needed: {input_path.name}")
            return False

    except Exception as e:
        print(f"✗ Error processing {input_path.name}: {e}")
        return False


def clean_summaries(input_dir, output_dir=None):
    """Clean all .txt files in the input directory."""

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return

    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return

    # Set output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        in_place = False
    else:
        output_path = input_path
        in_place = True

    # Find all .txt files
    txt_files = list(input_path.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in '{input_dir}'")
        return

    print(f"Found {len(txt_files)} .txt files to process")
    if in_place:
        print("Processing files in-place...")
    else:
        print(f"Output directory: {output_path}")
    print("-" * 50)

    cleaned_count = 0

    for txt_file in sorted(txt_files):
        output_file = output_path / txt_file.name
        if process_file(txt_file, output_file):
            cleaned_count += 1

    print("-" * 50)
    print(f"Processing complete!")
    print(f"Files processed: {len(txt_files)}")
    print(f"Files cleaned: {cleaned_count}")
    print(f"Files unchanged: {len(txt_files) - cleaned_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean summary files by removing <think> tags and their content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Clean files in-place
    python clean_summaries.py data/generated_summaries
    
    # Clean files to a new directory
    python clean_summaries.py data/generated_summaries data/cleaned_summaries
        """
    )

    parser.add_argument(
        "input_dir",
        help="Directory containing summary files to clean"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Output directory (optional, defaults to in-place cleaning)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview what would be cleaned without making changes"
    )

    args = parser.parse_args()

    if args.preview:
        print("PREVIEW MODE - No files will be modified")
        print("=" * 50)

        input_path = Path(args.input_dir)
        for txt_file in sorted(input_path.glob("*.txt")):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                if '<think>' in content.lower():
                    print(f"Would clean: {txt_file.name}")
                else:
                    print(f"No changes: {txt_file.name}")
            except Exception as e:
                print(f"Error reading {txt_file.name}: {e}")
    else:
        clean_summaries(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
