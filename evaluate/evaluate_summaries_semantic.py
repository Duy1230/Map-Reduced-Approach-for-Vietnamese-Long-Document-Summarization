import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np

# Import evaluation libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from rouge_score import rouge_scorer
    import bert_score
except ImportError as e:
    print(f"Error: Missing required package. Please install with:")
    print("pip install sentence-transformers scikit-learn rouge-score bert-score")
    sys.exit(1)

# DeepEval imports for LLM-based evaluation
try:
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.metrics import GEval
    from deepeval.models.base_model import DeepEvalBaseLLM
    from openai import OpenAI
except ImportError as e:
    print("Warning: DeepEval or OpenAI not available. LLM-based metrics will be skipped.")
    print("To use LLM-based metrics, install: pip install deepeval openai")

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    print("Warning: dotenv not available. Environment variables will be loaded from system.")


class OpenRouterModel(DeepEvalBaseLLM):
    """Custom model class for OpenRouter API integration with DeepEval."""

    def __init__(self, model_name: str = "openai/gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key must be provided either as parameter or OPENROUTER_API_KEY environment variable")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/deepeval",
                "X-Title": "DeepEval Semantic Evaluation"
            }
        )

        super().__init__(model_name)

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: Optional[type] = None, **kwargs) -> str:
        """Generate response using OpenRouter API."""
        try:
            messages = [{"role": "user", "content": prompt}]

            completion_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0),
                "max_tokens": kwargs.get("max_tokens", 4096),
            }

            if schema:
                if "openai/" in self.model_name:
                    completion_kwargs["response_format"] = {
                        "type": "json_object"}

                schema_instruction = f"\n\nYou must respond with a valid JSON object only, no other text. The JSON should match this structure: {schema.__annotations__ if hasattr(schema, '__annotations__') else str(schema)}"
                messages[0]["content"] += schema_instruction

            response = self.client.chat.completions.create(**completion_kwargs)
            content = response.choices[0].message.content

            if schema and content:
                if "```json" in content:
                    content = content.split("```json")[
                        1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                try:
                    if hasattr(schema, 'model_validate_json'):
                        return schema.model_validate_json(content)
                    elif hasattr(schema, 'parse_raw'):
                        return schema.parse_raw(content)
                    else:
                        print(
                            f"Warning: Schema {schema} does not have model_validate_json or parse_raw. Returning raw string.")
                        return content
                except Exception as parse_error:
                    error_message = (
                        f"Failed to parse LLM output into schema '{schema.__name__ if hasattr(schema, '__name__') else str(schema)}'.\n"
                        f"Error: {parse_error}\n"
                        f"LLM content that failed parsing: '{content}'"
                    )
                    print(error_message)
                    raise ValueError(error_message) from parse_error

            return content

        except Exception as e:
            print(f"Debug: Attempting to use model '{self.model_name}'")
            print(f"Debug: Full error: {str(e)}")
            raise Exception(f"Error generating response with OpenRouter: {e}")

    async def a_generate(self, prompt: str, schema: Optional[type] = None, **kwargs) -> str:
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self) -> str:
        return self.model_name


class SemanticEvaluator:
    """Comprehensive semantic evaluation for summarization tasks."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the evaluator with sentence transformer model."""
        print(f"Loading sentence transformer model: {embedding_model}")
        self.sentence_model = SentenceTransformer(embedding_model)
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts using sentence embeddings."""
        embeddings = self.sentence_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def compute_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores between generated and reference summaries."""
        scores = self.rouge_scorer.score(reference, generated)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure,
        }

    def compute_bert_score(self, generated: List[str], reference: List[str]) -> Dict[str, float]:
        """Compute BERTScore between generated and reference summaries."""
        try:
            P, R, F1 = bert_score.score(
                generated, reference, lang="vi", verbose=False)
            return {
                'bert_precision': float(P.mean()),
                'bert_recall': float(R.mean()),
                'bert_f1': float(F1.mean()),
            }
        except Exception as e:
            print(f"Warning: BERTScore computation failed: {e}")
            return {
                'bert_precision': 0.0,
                'bert_recall': 0.0,
                'bert_f1': 0.0,
            }

    def evaluate_pair(self, generated: str, reference: str) -> Dict[str, float]:
        """Evaluate a single generated-reference pair."""
        results = {}

        # Semantic similarity
        results['semantic_similarity'] = self.compute_semantic_similarity(
            generated, reference)

        # ROUGE scores
        rouge_scores = self.compute_rouge_scores(generated, reference)
        results.update(rouge_scores)

        return results


def load_texts_from_folder(folder_path: str, file_extension: str = ".txt") -> Dict[str, str]:
    """Load all text files from a folder and return as a dictionary."""
    texts = {}
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return texts

    for file_path in sorted(folder.glob(f"*{file_extension}")):
        if file_path.is_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts[file_path.name] = f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")

    return texts


def evaluate_with_llm_geval(generated_summaries: Dict[str, str],
                            ground_truth_summaries: Dict[str, str],
                            model_name: str = "openai/gpt-4o-mini",
                            use_openrouter: bool = False,
                            openrouter_api_key: Optional[str] = None,
                            max_samples: Optional[int] = None) -> Dict[str, float]:
    """Evaluate using DeepEval's G-Eval for comprehensive summarization assessment."""

    try:
        # Find common files
        common_files = set(generated_summaries.keys()) & set(
            ground_truth_summaries.keys())
        if not common_files:
            print("No common files found for LLM evaluation")
            return {
                'llm_correctness_mean': 0.0,
                'llm_correctness_std': 0.0,
                'llm_correctness_min': 0.0,
                'llm_correctness_max': 0.0,
                'llm_coherence_mean': 0.0,
                'llm_coherence_std': 0.0,
                'llm_coherence_min': 0.0,
                'llm_coherence_max': 0.0,
                'llm_successful_cases': 0,
                'llm_failed_cases': 0,
                'llm_total_cases_processed': 0,
                'llm_evaluation_failed': True,
                'llm_failure_reason': 'No common files found between generated and ground truth summaries'
            }

        sorted_files = sorted(common_files)
        if max_samples is not None:
            sorted_files = sorted_files[:max_samples]

        # Setup model
        if use_openrouter:
            evaluation_model = OpenRouterModel(
                model_name=model_name, api_key=openrouter_api_key)
        else:
            if not os.getenv("OPENAI_API_KEY"):
                print("Warning: OPENAI_API_KEY not set, skipping LLM evaluation")
                return {
                    'llm_correctness_mean': 0.0,
                    'llm_correctness_std': 0.0,
                    'llm_correctness_min': 0.0,
                    'llm_correctness_max': 0.0,
                    'llm_coherence_mean': 0.0,
                    'llm_coherence_std': 0.0,
                    'llm_coherence_min': 0.0,
                    'llm_coherence_max': 0.0,
                    'llm_successful_cases': 0,
                    'llm_failed_cases': 0,
                    'llm_total_cases_processed': 0,
                    'llm_evaluation_failed': True,
                    'llm_failure_reason': 'OPENAI_API_KEY environment variable not set'
                }
            evaluation_model = model_name

        # Create test cases for G-Eval evaluation
        test_cases = []
        for filename in sorted_files:
            # For summarization: input=original_text, expected_output=reference_summary, actual_output=generated_summary
            # Note: We don't have original text, so we'll use reference as context
            test_case = LLMTestCase(
                input="Evaluate the quality of this summary",
                actual_output=generated_summaries[filename],
                expected_output=ground_truth_summaries[filename]
            )
            test_cases.append(test_case)

        # Create G-Eval metrics for summarization
        # 1. Correctness/Faithfulness metric
        correctness_criteria = """
        Correctness (1-5): Measures how accurately the generated summary captures the key information and main points from the reference summary. 
        Criteria:
        - How much correct information does the generated summary contain compare to the reference summary?
        - Does the generated summay contains contradictions with the source document?
        - How well does the generated summary cover key points and main themes (or events) with respect to the reference?
        """

        correctness_metric = GEval(
            name="Summary Correctness",
            criteria=correctness_criteria,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT],
            model=evaluation_model
        )

        # 2. Coherence metric
        coherence_criteria = """
        Coherence (1-5): Measures the logical flow, structure, and organization of the generated summary.
        The summary should:
        - Have a clear and logical structure that flows from sentence to sentence
        - Be well-organized with coherent progression of ideas
        - Maintain consistency in style and tone throughout
        - Not be just a collection of random facts, but a cohesive narrative
        - Use appropriate transitions and connections between concepts
        """

        coherence_metric = GEval(
            name="Summary Coherence",
            criteria=coherence_criteria,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=evaluation_model
        )

        print(
            f"Running G-Eval (Correctness & Coherence) evaluation on {len(test_cases)} samples...")

        # Process test cases individually to avoid losing all scores when one fails
        correctness_scores = []
        coherence_scores = []
        failed_cases_count = 0
        successful_cases_count = 0

        for i, test_case in enumerate(test_cases):
            filename_info = sorted_files[i] if i < len(
                sorted_files) else f"Test Case Index {i}"
            try:
                print(f"  Processing {i+1}/{len(test_cases)}: {filename_info}")

                # Evaluate single test case with both metrics
                single_result = evaluate(
                    test_cases=[test_case], metrics=[correctness_metric, coherence_metric])

                if not single_result.test_results:
                    print(
                        f"    Warning: No test results for {filename_info}. Skipping.")
                    failed_cases_count += 1
                    continue

                test_result = single_result.test_results[0]

                if not test_result.metrics_data or len(test_result.metrics_data) < 2:
                    print(
                        f"    Warning: Insufficient metrics data for {filename_info}. Skipping.")
                    failed_cases_count += 1
                    continue

                # Extract scores for both metrics
                # First metric (Correctness)
                correctness_result = test_result.metrics_data[0]
                # Second metric (Coherence)
                coherence_result = test_result.metrics_data[1]

                # Validate and store correctness score
                if hasattr(correctness_result, 'score') and isinstance(correctness_result.score, (int, float)):
                    correctness_score = correctness_result.score
                else:
                    print(
                        f"    Warning: Invalid correctness score for {filename_info}. Skipping.")
                    failed_cases_count += 1
                    continue

                # Validate and store coherence score
                if hasattr(coherence_result, 'score') and isinstance(coherence_result.score, (int, float)):
                    coherence_score = coherence_result.score
                else:
                    print(
                        f"    Warning: Invalid coherence score for {filename_info}. Skipping.")
                    failed_cases_count += 1
                    continue

                # Both scores are valid
                correctness_scores.append(correctness_score)
                coherence_scores.append(coherence_score)
                successful_cases_count += 1
                print(
                    f"    ✓ Correctness: {correctness_score:.4f}, Coherence: {coherence_score:.4f}")

            except Exception as e:
                print(
                    f"    Error processing G-Eval for {filename_info}: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}. Skipping.")
                failed_cases_count += 1

        total_cases_processed = len(test_cases)

        if failed_cases_count > 0:
            print(
                f"Completed: Skipped {failed_cases_count}/{total_cases_processed} test cases due to errors.")

        if not correctness_scores or not coherence_scores:
            print("Warning: No valid G-Eval scores were collected.")
            return {
                'llm_correctness_mean': 0.0,
                'llm_correctness_std': 0.0,
                'llm_correctness_min': 0.0,
                'llm_correctness_max': 0.0,
                'llm_coherence_mean': 0.0,
                'llm_coherence_std': 0.0,
                'llm_coherence_min': 0.0,
                'llm_coherence_max': 0.0,
                'llm_successful_cases': 0,
                'llm_failed_cases': failed_cases_count,
                'llm_total_cases_processed': total_cases_processed
            }

        print(
            f"Successfully processed {successful_cases_count}/{total_cases_processed} test cases.")

        return {
            'llm_correctness_mean': float(np.mean(correctness_scores)),
            'llm_correctness_std': float(np.std(correctness_scores)),
            'llm_correctness_min': float(np.min(correctness_scores)),
            'llm_correctness_max': float(np.max(correctness_scores)),
            'llm_coherence_mean': float(np.mean(coherence_scores)),
            'llm_coherence_std': float(np.std(coherence_scores)),
            'llm_coherence_min': float(np.min(coherence_scores)),
            'llm_coherence_max': float(np.max(coherence_scores)),
            'llm_successful_cases': successful_cases_count,
            'llm_failed_cases': failed_cases_count,
            'llm_total_cases_processed': total_cases_processed
        }

    except Exception as e:
        print(f"Warning: LLM evaluation failed: {e}")
        return {
            'llm_correctness_mean': 0.0,
            'llm_correctness_std': 0.0,
            'llm_correctness_min': 0.0,
            'llm_correctness_max': 0.0,
            'llm_coherence_mean': 0.0,
            'llm_coherence_std': 0.0,
            'llm_coherence_min': 0.0,
            'llm_coherence_max': 0.0,
            'llm_successful_cases': 0,
            'llm_failed_cases': 0,
            'llm_total_cases_processed': 0,
            'llm_evaluation_failed': True,
            'llm_failure_reason': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated summaries using semantic similarity metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Basic semantic evaluation
    python evaluate_summaries_semantic.py data/generated_summaries data/summary
    
    # With LLM-based relevancy using OpenRouter
    export OPENROUTER_API_KEY=your_openrouter_key
    python evaluate_summaries_semantic.py data/generated_summaries data/summary --use-openrouter --include-llm-eval
    
    # Test with limited samples
    python evaluate_summaries_semantic.py data/generated_summaries data/summary --max-samples 10
        """
    )

    parser.add_argument(
        "generated_summaries_dir",
        help="Directory containing generated summaries"
    )
    parser.add_argument(
        "ground_truth_summaries_dir",
        help="Directory containing ground truth summaries"
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for semantic similarity (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--include-llm-eval",
        action="store_true",
        help="Include G-Eval LLM-based evaluation (Correctness & Coherence)"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="Model to use for G-Eval evaluation (default: openai/gpt-4o-mini)"
    )
    parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Use OpenRouter API for G-Eval evaluation"
    )
    parser.add_argument(
        "--openrouter-api-key",
        help="OpenRouter API key (can also use OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate (useful for testing)"
    )
    parser.add_argument(
        "--output",
        help="Output file to save detailed results (JSON format)"
    )

    args = parser.parse_args()

    # Validate directories
    for dir_path, name in [(args.generated_summaries_dir, "generated summaries"),
                           (args.ground_truth_summaries_dir, "ground truth summaries")]:
        if not os.path.exists(dir_path):
            print(
                f"Error: {name.title()} directory '{dir_path}' does not exist")
            sys.exit(1)

    print("Semantic Summarization Evaluation")
    print("=" * 50)
    print(f"Generated summaries directory: {args.generated_summaries_dir}")
    print(
        f"Ground truth summaries directory: {args.ground_truth_summaries_dir}")
    print(f"Embedding model: {args.embedding_model}")
    if args.include_llm_eval:
        print(
            f"G-Eval: {'OpenRouter' if args.use_openrouter else 'OpenAI'} - {args.model}")
    print()

    # Load data
    print("Loading files...")
    print("-" * 50)

    generated_summaries = load_texts_from_folder(args.generated_summaries_dir)
    ground_truth_summaries = load_texts_from_folder(
        args.ground_truth_summaries_dir)

    if not generated_summaries or not ground_truth_summaries:
        print("Error: Could not load summaries from one or both directories")
        sys.exit(1)

    print(f"Loaded {len(generated_summaries)} generated summaries")
    print(f"Loaded {len(ground_truth_summaries)} ground truth summaries")

    # Find common files
    common_files = set(generated_summaries.keys()) & set(
        ground_truth_summaries.keys())
    if not common_files:
        print("Error: No matching files found between directories")
        sys.exit(1)

    sorted_files = sorted(common_files)
    if args.max_samples is not None:
        sorted_files = sorted_files[:args.max_samples]
        print(f"Limiting evaluation to first {len(sorted_files)} samples")

    print(f"Evaluating {len(sorted_files)} file pairs")

    # Initialize evaluator
    print("\nInitializing semantic evaluator...")
    print("-" * 50)
    evaluator = SemanticEvaluator(embedding_model=args.embedding_model)

    # Run evaluation
    print("\nRunning semantic evaluation...")
    print("-" * 50)

    all_results = []
    semantic_similarities = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for i, filename in enumerate(sorted_files):
        generated = generated_summaries[filename]
        reference = ground_truth_summaries[filename]

        result = evaluator.evaluate_pair(generated, reference)
        result['filename'] = filename
        all_results.append(result)

        semantic_similarities.append(result['semantic_similarity'])
        rouge1_scores.append(result['rouge1_f'])
        rouge2_scores.append(result['rouge2_f'])
        rougeL_scores.append(result['rougeL_f'])

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(sorted_files)} files...")

    # Compute BERTScore for all pairs
    print("Computing BERTScore...")
    generated_texts = [generated_summaries[f] for f in sorted_files]
    reference_texts = [ground_truth_summaries[f] for f in sorted_files]
    bert_scores = evaluator.compute_bert_score(
        generated_texts, reference_texts)

    # LLM-based evaluation using G-Eval
    llm_scores = {}
    if args.include_llm_eval:
        print("Running G-Eval (Correctness & Coherence) evaluation...")
        llm_scores = evaluate_with_llm_geval(
            generated_summaries, ground_truth_summaries,
            model_name=args.model,
            use_openrouter=args.use_openrouter,
            openrouter_api_key=args.openrouter_api_key,
            max_samples=args.max_samples
        )

    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)

    print(f"Semantic Similarity (Sentence Transformers):")
    print(f"  Mean: {np.mean(semantic_similarities):.4f}")
    print(f"  Std:  {np.std(semantic_similarities):.4f}")
    print(f"  Min:  {np.min(semantic_similarities):.4f}")
    print(f"  Max:  {np.max(semantic_similarities):.4f}")

    print(f"\nROUGE Scores:")
    print(
        f"  ROUGE-1 F1: {np.mean(rouge1_scores):.4f} (±{np.std(rouge1_scores):.4f})")
    print(
        f"  ROUGE-2 F1: {np.mean(rouge2_scores):.4f} (±{np.std(rouge2_scores):.4f})")
    print(
        f"  ROUGE-L F1: {np.mean(rougeL_scores):.4f} (±{np.std(rougeL_scores):.4f})")

    print(f"\nBERTScore:")
    print(f"  Precision: {bert_scores['bert_precision']:.4f}")
    print(f"  Recall:    {bert_scores['bert_recall']:.4f}")
    print(f"  F1:        {bert_scores['bert_f1']:.4f}")

    if llm_scores:
        print(f"\nG-Eval Results:")
        if llm_scores.get('llm_evaluation_failed', False):
            print(f"  Status: FAILED")
            print(
                f"  Reason: {llm_scores.get('llm_failure_reason', 'Unknown error')}")
            print(f"  Note: G-Eval evaluation was attempted but failed completely.")
        elif llm_scores.get('llm_successful_cases', 0) == 0:
            print(f"  Status: NO VALID SCORES")
            print(
                f"  Total cases processed: {llm_scores.get('llm_total_cases_processed', 0)}")
            print(f"  Failed cases: {llm_scores.get('llm_failed_cases', 0)}")
            print(f"  Note: G-Eval evaluation ran but no valid scores were obtained.")
        else:
            print(f"  Correctness:")
            print(f"    Mean: {llm_scores['llm_correctness_mean']:.4f}")
            print(f"    Std:  {llm_scores['llm_correctness_std']:.4f}")
            print(f"    Min:  {llm_scores['llm_correctness_min']:.4f}")
            print(f"    Max:  {llm_scores['llm_correctness_max']:.4f}")

            print(f"  Coherence:")
            print(f"    Mean: {llm_scores['llm_coherence_mean']:.4f}")
            print(f"    Std:  {llm_scores['llm_coherence_std']:.4f}")
            print(f"    Min:  {llm_scores['llm_coherence_min']:.4f}")
            print(f"    Max:  {llm_scores['llm_coherence_max']:.4f}")

            total_processed = llm_scores.get('llm_total_cases_processed', 0)
            successful = llm_scores.get('llm_successful_cases', 0)
            failed = llm_scores.get('llm_failed_cases', 0)

            if total_processed > 0:
                print(
                    f"  Cases: {successful}/{total_processed} successful ({successful/total_processed*100:.1f}%)")
                if failed > 0:
                    print(
                        f"  Failed: {failed} cases were skipped due to errors")

    # Summary statistics
    print(f"\nSummary:")
    print("-" * 50)

    # Define thresholds for interpretation
    high_sim_count = sum(1 for s in semantic_similarities if s >= 0.7)
    med_sim_count = sum(1 for s in semantic_similarities if 0.4 <= s < 0.7)
    low_sim_count = sum(1 for s in semantic_similarities if s < 0.4)

    print(f"Semantic Similarity Distribution:")
    print(
        f"  High similarity (>=0.7): {high_sim_count}/{len(sorted_files)} ({high_sim_count/len(sorted_files)*100:.1f}%)")
    print(
        f"  Medium similarity (0.4-0.7): {med_sim_count}/{len(sorted_files)} ({med_sim_count/len(sorted_files)*100:.1f}%)")
    print(
        f"  Low similarity (<0.4): {low_sim_count}/{len(sorted_files)} ({low_sim_count/len(sorted_files)*100:.1f}%)")

    # Save detailed results if requested
    if args.output:
        output_data = {
            'summary_statistics': {
                'semantic_similarity': {
                    'mean': float(np.mean(semantic_similarities)),
                    'std': float(np.std(semantic_similarities)),
                    'min': float(np.min(semantic_similarities)),
                    'max': float(np.max(semantic_similarities))
                },
                'rouge_scores': {
                    'rouge1_f1': float(np.mean(rouge1_scores)),
                    'rouge2_f1': float(np.mean(rouge2_scores)),
                    'rougeL_f1': float(np.mean(rougeL_scores))
                },
                'bert_scores': bert_scores,
                'llm_scores': llm_scores
            },
            'detailed_results': all_results
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
