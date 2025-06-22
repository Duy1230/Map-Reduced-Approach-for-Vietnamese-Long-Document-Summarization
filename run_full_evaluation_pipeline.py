from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from runners.run_summarization_ollama_mapreduce import create_map_reduce_graph, summarize_document_mapreduce
from runners.run_summarization_ollama_mapreduce_critique import (
    create_map_reduce_graph_with_critique,
    create_critique_refine_graph,
    summarize_document_mapreduce_with_critique
)
from runners.run_summarization_ollama_iterative import create_iterative_refinement_graph, summarize_document_iterative
from runners.run_summarization_ollama import generate_summary_ollama
from runners.run_summarization_ollama_mapreduce_hierarchical import hierarchical_summarize_document
from langchain_core.language_models.llms import LLM
import requests
import os
import sys
import json
import copy
import time
import logging
import asyncio
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import traceback

# Add current directory to path to import our modules
sys.path.append('.')

# Import our custom modules


def clean_thinking_tokens(text: str) -> str:
    """
    Remove thinking tokens from model output.
    Removes everything between <think> and </think> tags, including the tags themselves.
    Also handles variations like <thinking>, <thought>, etc.
    """
    if not text:
        return text

    # Pattern to match thinking tags (case insensitive)
    patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<thought>.*?</thought>',
        r'<reasoning>.*?</reasoning>',
        r'<analysis>.*?</analysis>'
    ]

    cleaned_text = text
    for pattern in patterns:
        # Use DOTALL flag to match across newlines
        cleaned_text = re.sub(pattern, '', cleaned_text,
                              flags=re.DOTALL | re.IGNORECASE)

    # Clean up extra whitespace that might be left
    # Multiple newlines to double
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()

    return cleaned_text


class OllamaLLM(LLM):
    """Custom LLM wrapper for Ollama API with thinking token cleaning"""

    ollama_url: str
    model_name: str
    max_new_tokens: int = 2048

    def __init__(self, ollama_url: str, model_name: str, max_new_tokens: int = 2048):
        super().__init__(
            ollama_url=ollama_url,
            model_name=model_name,
            max_new_tokens=max_new_tokens
        )

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_new_tokens
            }
        }

        resp = requests.post(f"{self.ollama_url}/api/generate", json=payload)
        resp.raise_for_status()

        # Get the raw response
        raw_response = resp.json()["response"]

        # Clean thinking tokens before returning
        cleaned_response = clean_thinking_tokens(raw_response)

        # Log if thinking tokens were removed (for debugging)
        if len(cleaned_response) < len(raw_response):
            tokens_removed = len(raw_response) - len(cleaned_response)
            if hasattr(self, '_logger'):
                self._logger.debug(
                    f"Removed {tokens_removed} characters of thinking tokens")

        return cleaned_response

    async def _acall(self, prompt: str, stop=None, run_manager=None, **kwargs):
        return self._call(prompt, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def get_num_tokens(self, text: str) -> int:
        # Simple approximation - could be improved with proper tokenizer
        return len(text.split())


class PipelineRunner:
    """Manages the complete evaluation pipeline with comprehensive logging."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.start_time = datetime.now()
        self.results = {}

        # Setup logging
        self.setup_logging()

        # Create output directories
        self.setup_directories()

        # Log initial configuration
        self.log_configuration()

    def setup_logging(self):
        """Setup comprehensive logging system."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Create timestamped log file
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_run_{timestamp}.log"

        # Setup logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.log_file = log_file

        self.logger.info("="*80)
        self.logger.info(
            f"STARTING {self.config.get('approach', 'mapreduce').upper()} SUMMARIZATION EVALUATION PIPELINE")
        self.logger.info("="*80)
        self.logger.info(f"Log file: {log_file}")

    def setup_directories(self):
        """Create necessary output directories."""
        approach = self.config.get('approach', 'mapreduce')

        for model in self.config['models']:
            model_name_safe = model.replace(':', '_').replace('.', '_')
            summary_dir = Path(
                f"{self.config['generated_summaries_dir']}_{approach}_{model_name_safe}")
            summary_dir.mkdir(parents=True, exist_ok=True)

        Path("evaluation_results").mkdir(exist_ok=True)

    def log_configuration(self):
        """Log the current configuration."""
        self.logger.info("\nCONFIGURATION:")
        self.logger.info("-" * 50)

        config_log = json.dumps(self.config, indent=2)
        for line in config_log.split('\n'):
            self.logger.info(f"  {line}")

        # Log system information
        self.logger.info(f"\nSYSTEM INFORMATION:")
        self.logger.info("-" * 50)
        self.logger.info(f"  Python version: {sys.version}")
        self.logger.info(f"  Working directory: {os.getcwd()}")
        self.logger.info(f"  Timestamp: {self.start_time.isoformat()}")

        # Test thinking token cleaning (for verification)
        test_text = "This is a summary. <think>This is thinking content that should be removed.</think> This is the rest of the summary."
        cleaned_test = clean_thinking_tokens(test_text)
        self.logger.info(
            f"  Thinking token cleaning: {'✓ ENABLED' if len(cleaned_test) < len(test_text) else '✗ FAILED'}")

    def check_ollama_status(self) -> bool:
        """Check if Ollama server is running and models are available."""
        self.logger.info("\nCHECKING OLLAMA STATUS:")
        self.logger.info("-" * 50)

        try:
            import requests
            # Check if Ollama server is responding
            response = requests.get(
                f"{self.config['ollama_url']}/api/tags", timeout=10)
            response.raise_for_status()

            available_models = response.json()
            model_names = [model['name']
                           for model in available_models.get('models', [])]

            self.logger.info(f"  Ollama server: RUNNING")
            self.logger.info(f"  Available models: {model_names}")

            # Check if our required models are available
            missing_models = []
            for model in self.config['models']:
                if model not in model_names:
                    missing_models.append(model)

            if missing_models:
                self.logger.error(f"  Missing models: {missing_models}")
                return False

            self.logger.info(f"  All required models available: ✓")
            return True

        except Exception as e:
            self.logger.error(f"  Ollama server check failed: {e}")
            return False

    def count_documents(self) -> Dict[str, Any]:
        """Count and analyze the documents to be processed."""
        self.logger.info("\nANALYZING INPUT DOCUMENTS:")
        self.logger.info("-" * 50)

        docs_dir = Path(self.config['docs_dir'])
        summary_dir = Path(self.config['summary_dir'])

        doc_files = list(docs_dir.glob("*.txt")) if docs_dir.exists() else []
        ref_files = list(summary_dir.glob("*.txt")
                         ) if summary_dir.exists() else []

        # Find matching pairs
        doc_names = {f.name for f in doc_files}
        ref_names = {f.name for f in ref_files}
        matching_files = doc_names & ref_names

        # Analyze document sizes
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b")
        doc_info = []
        total_tokens = 0

        for filename in sorted(matching_files):
            doc_path = docs_dir / filename
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tokens = tokenizer.encode(content)
            token_count = len(tokens)
            total_tokens += token_count

            # Get appropriate chunk size based on approach
        approach = self.config.get('approach', 'mapreduce')
        if approach == 'mapreduce':
            chunk_size = self.config.get('chunk_size', 12000)
        elif approach == 'iterative':
            chunk_size = self.config.get('iterative_chunk_size', 12000)
        elif approach == 'truncated':
            chunk_size = self.config.get(
                'max_context', 16384) - self.config.get('max_new_tokens', 2048)
        elif approach == 'mapreduce_hierarchical':
            chunk_size = self.config.get('chunk_size', 12000)
        else:
            chunk_size = 12000  # fallback

            doc_info.append({
                'filename': filename,
                'char_count': len(content),
                'token_count': token_count,
                'estimated_chunks': max(1, token_count // chunk_size)
            })

        stats = {
            'total_documents': len(doc_files),
            'total_references': len(ref_files),
            'matching_pairs': len(matching_files),
            'total_tokens': total_tokens,
            'avg_tokens_per_doc': total_tokens / len(matching_files) if matching_files else 0,
            'documents': doc_info
        }

        self.logger.info(f"  Total documents: {stats['total_documents']}")
        self.logger.info(f"  Total references: {stats['total_references']}")
        self.logger.info(f"  Matching pairs: {stats['matching_pairs']}")
        self.logger.info(f"  Total tokens: {stats['total_tokens']:,}")
        self.logger.info(
            f"  Average tokens per document: {stats['avg_tokens_per_doc']:.0f}")

        # Log chunking estimates
        total_estimated_chunks = sum(
            doc['estimated_chunks'] for doc in doc_info)

        # Get chunk size for logging
        approach = self.config.get('approach', 'mapreduce')
        if approach == 'mapreduce':
            display_chunk_size = self.config.get('chunk_size', 12000)
        elif approach == 'iterative':
            display_chunk_size = self.config.get('iterative_chunk_size', 12000)
        elif approach == 'truncated':
            display_chunk_size = self.config.get(
                'max_context', 16384) - self.config.get('max_new_tokens', 2048)
        else:
            display_chunk_size = 12000

        self.logger.info(
            f"  Estimated total chunks (chunk_size={display_chunk_size}): {total_estimated_chunks}")

        return stats

    async def run_summarization_for_model(self, model_name: str) -> Dict[str, Any]:
        """Run summarization for a specific model using the specified approach."""
        model_name_safe = model_name.replace(':', '_').replace('.', '_')
        approach = self.config.get('approach', 'mapreduce')

        self.logger.info(
            f"\nRUNNING {approach.upper()} SUMMARIZATION - MODEL: {model_name}")
        self.logger.info("=" * 80)

        start_time = time.time()

        try:
            # Initialize components
            llm = OllamaLLM(
                ollama_url=self.config['ollama_url'],
                model_name=model_name,
                max_new_tokens=self.config['max_new_tokens']
            )

            # Initialize tokenizer for accurate token-based splitting
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-3b")

            # Create custom length function using the actual tokenizer
            def length_function(text: str) -> int:
                return len(tokenizer.encode(text))

            # Setup approach-specific components
            app = None
            text_splitter = None

            if approach == 'mapreduce':
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config['chunk_size'],
                    chunk_overlap=self.config['chunk_overlap'],
                    length_function=length_function,
                    separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
                )
                app = create_map_reduce_graph(
                    llm, token_max=self.config['token_max'])

            elif approach == 'mapreduce_critique':
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config['chunk_size'],
                    chunk_overlap=self.config['chunk_overlap'],
                    length_function=length_function,
                    separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
                )
                # Create enhanced map-reduce graph with integrated critique
                mapreduce_app = create_map_reduce_graph_with_critique(
                    llm,
                    token_max=self.config['token_max'],
                    max_critique_iterations=self.config.get(
                        'max_critique_iterations', 2)
                )
                critique_app = create_critique_refine_graph(
                    llm)  # Dummy for compatibility
                # Store both in a tuple for later use
                app = (mapreduce_app, critique_app)

            elif approach == 'iterative':
                # For iterative refinement, use larger chunks with more overlap
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.get('iterative_chunk_size', 8000),
                    chunk_overlap=self.config.get(
                        'iterative_chunk_overlap', 400),
                    length_function=length_function,
                    separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
                )
                app = create_iterative_refinement_graph(llm)

            elif approach == 'truncated':
                # For truncated approach, we don't need text splitter or app
                pass
            elif approach == 'mapreduce_hierarchical':
                # For hierarchical approach, we need to load the tree structure
                # This will be handled per-document in the processing loop
                pass
            else:
                raise ValueError(f"Unknown approach: {approach}")

            # Setup directories
            docs_dir = self.config['docs_dir']
            summary_dir = self.config['summary_dir']
            generated_summaries_dir = f"{self.config['generated_summaries_dir']}_{approach}_{model_name_safe}"

            os.makedirs(generated_summaries_dir, exist_ok=True)

            # Process documents
            generated_summaries = []
            references = []
            processing_stats = []

            for fname in sorted(os.listdir(docs_dir)):
                doc_path = os.path.join(docs_dir, fname)
                ref_path = os.path.join(summary_dir, fname)
                gen_path = os.path.join(generated_summaries_dir, fname)

                # Skip if already exists
                if os.path.isfile(gen_path):
                    self.logger.info(
                        f"  {fname}: Already exists, loading for evaluation")
                    with open(gen_path, 'r', encoding='utf-8') as f:
                        generated_summaries.append(f.read())
                    if os.path.isfile(ref_path):
                        with open(ref_path, 'r', encoding='utf-8') as f:
                            references.append(f.read())
                    continue

                if not os.path.isfile(ref_path):
                    self.logger.warning(
                        f"  {fname}: No reference summary, skipping")
                    continue

                # Process document
                doc_start_time = time.time()

                with open(doc_path, 'r', encoding='utf-8') as f:
                    doc_text = f.read()
                with open(ref_path, 'r', encoding='utf-8') as f:
                    ref = f.read()

                # Get document stats
                doc_tokens = tokenizer.encode(doc_text)

                # Generate summary based on approach
                if approach == 'mapreduce':
                    doc_chunks = text_splitter.split_text(doc_text)
                    chunk_count = len(doc_chunks)
                    self.logger.info(
                        f"  {fname}: {len(doc_tokens)} tokens → {chunk_count} chunks (avg {len(doc_tokens)//chunk_count if chunk_count > 0 else 0} tokens/chunk)")

                    summary = await summarize_document_mapreduce(doc_text, app, text_splitter)

                elif approach == 'mapreduce_critique':
                    doc_chunks = text_splitter.split_text(doc_text)
                    chunk_count = len(doc_chunks)
                    self.logger.info(
                        f"  {fname}: {len(doc_tokens)} tokens → {chunk_count} chunks + critique refinement")

                    # Unpack the two apps
                    mapreduce_app, critique_app = app
                    summary = await summarize_document_mapreduce_with_critique(
                        doc_text, mapreduce_app, critique_app, text_splitter,
                        max_critique_iterations=self.config.get(
                            'max_critique_iterations', 3)
                    )

                elif approach == 'iterative':
                    doc_chunks = text_splitter.split_text(doc_text)
                    chunk_count = len(doc_chunks)
                    self.logger.info(
                        f"  {fname}: {len(doc_tokens)} tokens → {chunk_count} chunks for iterative refinement")

                    summary = await summarize_document_iterative(doc_text, app, text_splitter)

                elif approach == 'truncated':
                    # Truncated approach - use the existing function
                    max_context = self.config.get('max_context', 16384)
                    max_input_tokens = max_context - \
                        self.config['max_new_tokens']

                    chunk_count = 1  # No chunking for truncated
                    if len(doc_tokens) > max_input_tokens:
                        self.logger.info(
                            f"  {fname}: {len(doc_tokens)} tokens → truncated to {max_input_tokens} tokens")
                    else:
                        self.logger.info(
                            f"  {fname}: {len(doc_tokens)} tokens (no truncation needed)")

                    summary = generate_summary_ollama(
                        doc_text, tokenizer, max_input_tokens,
                        self.config['max_new_tokens'],
                        self.config['ollama_url'], model_name
                    )

                elif approach == 'mapreduce_hierarchical':
                    # Hierarchical approach - find the document in the tree structure
                    tree_json_path = self.config.get(
                        'tree_json_path', 'data_1/document_tree.json')

                    try:
                        with open(tree_json_path, 'r', encoding='utf-8') as f:
                            tree = json.load(f)
                    except FileNotFoundError:
                        self.logger.error(
                            f"  {fname}: Tree file not found at {tree_json_path}")
                        continue
                    except json.JSONDecodeError:
                        self.logger.error(
                            f"  {fname}: Invalid JSON in tree file {tree_json_path}")
                        continue

                    # Find the document node in the tree by matching filename
                    doc_name_base = os.path.splitext(
                        fname)[0]  # Remove .txt extension
                    document_node = None

                    for doc_node in tree.get('children', []):
                        if doc_node.get('type') == 'Document' and doc_node.get('text', '') == doc_name_base:
                            document_node = doc_node
                            break

                    if document_node is None:
                        self.logger.warning(
                            f"  {fname}: Document '{doc_name_base}' not found in tree structure, skipping")
                        continue

                    # Count total paragraphs and headers in the document
                    def count_nodes(node, node_type):
                        count = 1 if node.get('type') == node_type else 0
                        for child in node.get('children', []):
                            count += count_nodes(child, node_type)
                        return count

                    header_count = count_nodes(document_node, 'Header')
                    paragraph_count = count_nodes(document_node, 'Paragraph')
                    chunk_count = header_count  # Hierarchical chunks are headers that get collapsed

                    self.logger.info(
                        f"  {fname}: {len(doc_tokens)} tokens → hierarchical structure "
                        f"({header_count} headers, {paragraph_count} paragraphs, max_depth={self.config.get('max_depth', 2)})")

                    # Create a deep copy of the document node to avoid modifying the original tree
                    doc_copy = copy.deepcopy(document_node)

                    # Run hierarchical summarization
                    summary = await hierarchical_summarize_document(
                        doc_copy,
                        max_depth=self.config.get('max_depth', 2),
                        llm=llm,
                        chunk_size=self.config.get('chunk_size', 12000),
                        chunk_overlap=self.config.get('chunk_overlap', 200)
                    )

                # Additional cleaning to ensure no thinking tokens remain
                original_summary_length = len(summary)
                summary = clean_thinking_tokens(summary)

                if len(summary) < original_summary_length:
                    tokens_removed = original_summary_length - len(summary)
                    self.logger.info(
                        f"    Cleaned {tokens_removed} characters of thinking tokens from final summary")

                # Save summary
                with open(gen_path, 'w', encoding='utf-8') as f:
                    f.write(summary)

                doc_end_time = time.time()
                doc_processing_time = doc_end_time - doc_start_time

                processing_stats.append({
                    'filename': fname,
                    'original_tokens': len(doc_tokens),
                    'chunk_count': chunk_count,
                    'processing_time': doc_processing_time,
                    'summary_length': len(summary),
                    'approach': approach
                })

                self.logger.info(
                    f"  {fname}: Completed in {doc_processing_time:.1f}s")

                generated_summaries.append(summary)
                references.append(ref)

            end_time = time.time()
            total_time = end_time - start_time

            # Calculate statistics
            total_docs = len(processing_stats)
            total_original_tokens = sum(
                stat['original_tokens'] for stat in processing_stats)
            total_chunks = sum(stat['chunk_count']
                               for stat in processing_stats)
            avg_processing_time = sum(
                stat['processing_time'] for stat in processing_stats) / total_docs if total_docs > 0 else 0

            result = {
                'model_name': model_name,
                'approach': approach,
                'status': 'completed',
                'total_time': total_time,
                'total_documents': total_docs,
                'total_original_tokens': total_original_tokens,
                'total_chunks': total_chunks,
                'avg_processing_time_per_doc': avg_processing_time,
                'generated_summaries_count': len(generated_summaries),
                'references_count': len(references),
                'output_directory': generated_summaries_dir,
                'processing_details': processing_stats
            }

            self.logger.info(
                f"\n{approach.upper()} SUMMARIZATION COMPLETED - {model_name}")
            self.logger.info(f"  Total time: {total_time:.1f}s")
            self.logger.info(f"  Documents processed: {total_docs}")
            self.logger.info(f"  Total chunks generated: {total_chunks}")
            self.logger.info(
                f"  Average time per document: {avg_processing_time:.1f}s")

            return result

        except Exception as e:
            error_msg = f"Summarization failed for {model_name}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())

            return {
                'model_name': model_name,
                'approach': approach,
                'status': 'failed',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }

    def run_evaluation_for_model(self, model_name: str, summary_dir: str) -> Dict[str, Any]:
        """Run semantic evaluation for a specific model."""
        self.logger.info(f"\nRUNNING EVALUATION - MODEL: {model_name}")
        self.logger.info("=" * 80)

        start_time = time.time()

        try:
            # Run the evaluation script
            cmd = [
                sys.executable, "evaluate/evaluate_summaries_semantic.py",
                summary_dir,
                self.config['summary_dir'],
                "--embedding-model", self.config['evaluation']['embedding_model'],
                "--output", f"evaluation_results/{model_name.replace(':', '_').replace('.', '_')}_results.json"
            ]

            if self.config['evaluation']['include_llm_eval']:
                cmd.append("--include-llm-eval")
                if self.config['evaluation'].get('use_openrouter', False):
                    cmd.append("--use-openrouter")
                if self.config['evaluation'].get('llm_model'):
                    cmd.extend(
                        ["--model", self.config['evaluation']['llm_model']])

            if self.config['evaluation'].get('max_samples'):
                cmd.extend(
                    ["--max-samples", str(self.config['evaluation']['max_samples'])])

            self.logger.info(f"  Running command: {' '.join(cmd)}")

            # Set environment variables to handle encoding properly
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            # Run evaluation with proper encoding handling
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, encoding='utf-8', env=env)
            except UnicodeDecodeError:
                # Fallback to system default encoding on Windows
                result = subprocess.run(
                    cmd, capture_output=True, text=True, encoding=None, env=env)

            end_time = time.time()
            eval_time = end_time - start_time

            if result.returncode == 0:
                self.logger.info(
                    f"  Evaluation completed successfully in {eval_time:.1f}s")

                # Parse output for key metrics
                output_lines = result.stdout.split('\n')
                metrics = self.parse_evaluation_output(output_lines)

                return {
                    'model_name': model_name,
                    'status': 'completed',
                    'evaluation_time': eval_time,
                    'metrics': metrics,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                error_msg = f"Evaluation failed with return code {result.returncode}"
                self.logger.error(f"  {error_msg}")
                self.logger.error(f"  STDOUT: {result.stdout}")
                self.logger.error(f"  STDERR: {result.stderr}")

                return {
                    'model_name': model_name,
                    'status': 'failed',
                    'error': error_msg,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }

        except Exception as e:
            error_msg = f"Evaluation failed for {model_name}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())

            return {
                'model_name': model_name,
                'status': 'failed',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }

    def parse_evaluation_output(self, output_lines: List[str]) -> Dict[str, float]:
        """Parse evaluation script output to extract key metrics."""
        metrics = {}

        for i, original_line in enumerate(output_lines):
            line = original_line.strip()

            # Check context around current line for semantic similarity
            if "Mean:" in line:
                # Look for "Semantic Similarity" in nearby lines
                context_start = max(0, i-3)
                context_end = min(len(output_lines), i+2)
                context_lines = output_lines[context_start:context_end]

                if any("Semantic Similarity" in context_line for context_line in context_lines):
                    try:
                        metrics['semantic_similarity_mean'] = float(
                            line.split("Mean:")[1].strip())
                    except:
                        pass

            elif "ROUGE-1 F1:" in line:
                try:
                    metrics['rouge1_f1'] = float(
                        line.split("ROUGE-1 F1:")[1].split()[0])
                except:
                    pass

            elif "ROUGE-2 F1:" in line:
                try:
                    metrics['rouge2_f1'] = float(
                        line.split("ROUGE-2 F1:")[1].split()[0])
                except:
                    pass

            elif "ROUGE-L F1:" in line:
                try:
                    metrics['rougeL_f1'] = float(
                        line.split("ROUGE-L F1:")[1].split()[0])
                except:
                    pass

            elif "F1:" in line:
                # Look for "BERTScore" in nearby lines
                context_start = max(0, i-3)
                context_end = min(len(output_lines), i+2)
                context_lines = output_lines[context_start:context_end]

                if any("BERTScore" in context_line for context_line in context_lines):
                    try:
                        metrics['bert_f1'] = float(
                            line.split("F1:")[1].strip())
                    except:
                        pass

        return metrics

    async def run_full_pipeline(self):
        """Run the complete evaluation pipeline."""
        try:
            # Check Ollama status
            if not self.check_ollama_status():
                self.logger.error("Ollama server not ready. Exiting.")
                return

            # Analyze documents
            doc_stats = self.count_documents()
            self.results['document_stats'] = doc_stats

            if doc_stats['matching_pairs'] == 0:
                self.logger.error("No matching document pairs found. Exiting.")
                return

            # Run summarization for each model
            summarization_results = {}
            for model_name in self.config['models']:
                result = await self.run_summarization_for_model(model_name)
                summarization_results[model_name] = result

                if result['status'] != 'completed':
                    self.logger.warning(
                        f"Skipping evaluation for {model_name} due to summarization failure")
                    continue

            self.results['summarization'] = summarization_results

            # Run evaluation for each model
            evaluation_results = {}
            for model_name in self.config['models']:
                if summarization_results[model_name]['status'] == 'completed':
                    model_name_safe = model_name.replace(
                        ':', '_').replace('.', '_')
                    approach = self.config.get('approach', 'mapreduce')
                    summary_dir = f"{self.config['generated_summaries_dir']}_{approach}_{model_name_safe}"

                    result = self.run_evaluation_for_model(
                        model_name, summary_dir)
                    evaluation_results[model_name] = result

            self.results['evaluation'] = evaluation_results

            # Generate summary report
            self.generate_summary_report()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.logger.error(traceback.format_exc())

        finally:
            # Save final results
            self.save_final_results()

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        self.logger.info("\n" + "="*80)
        self.logger.info("FINAL SUMMARY REPORT")
        self.logger.info("="*80)

        end_time = datetime.now()
        total_pipeline_time = (end_time - self.start_time).total_seconds()

        self.logger.info(f"\nPIPELINE TIMING:")
        self.logger.info("-" * 50)
        self.logger.info(
            f"  Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(
            f"  End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(
            f"  Total duration: {total_pipeline_time:.1f}s ({total_pipeline_time/60:.1f} minutes)")

        # Summarization results
        self.logger.info(f"\nSUMMARIZATION RESULTS:")
        self.logger.info("-" * 50)

        for model_name, result in self.results.get('summarization', {}).items():
            if result['status'] == 'completed':
                self.logger.info(f"  {model_name}: ✓ COMPLETED")
                self.logger.info(f"    Documents: {result['total_documents']}")
                self.logger.info(f"    Total chunks: {result['total_chunks']}")
                self.logger.info(f"    Time: {result['total_time']:.1f}s")
                self.logger.info(
                    f"    Avg time/doc: {result['avg_processing_time_per_doc']:.1f}s")
            else:
                self.logger.info(
                    f"  {model_name}: ✗ FAILED - {result.get('error', 'Unknown error')}")

        # Evaluation results
        self.logger.info(f"\nEVALUATION RESULTS:")
        self.logger.info("-" * 50)

        comparison_data = []

        for model_name, result in self.results.get('evaluation', {}).items():
            if result['status'] == 'completed':
                metrics = result.get('metrics', {})
                self.logger.info(f"  {model_name}: ✓ COMPLETED")
                self.logger.info(
                    f"    Semantic Similarity: {metrics.get('semantic_similarity_mean', 'N/A'):.4f}")
                self.logger.info(
                    f"    ROUGE-1 F1: {metrics.get('rouge1_f1', 'N/A'):.4f}")
                self.logger.info(
                    f"    ROUGE-2 F1: {metrics.get('rouge2_f1', 'N/A'):.4f}")
                self.logger.info(
                    f"    ROUGE-L F1: {metrics.get('rougeL_f1', 'N/A'):.4f}")
                self.logger.info(
                    f"    BERT F1: {metrics.get('bert_f1', 'N/A'):.4f}")

                comparison_data.append({
                    'model': model_name,
                    'semantic_sim': metrics.get('semantic_similarity_mean', 0),
                    'rouge1': metrics.get('rouge1_f1', 0),
                    'rouge2': metrics.get('rouge2_f1', 0),
                    'rougeL': metrics.get('rougeL_f1', 0),
                    'bert_f1': metrics.get('bert_f1', 0)
                })
            else:
                self.logger.info(
                    f"  {model_name}: ✗ FAILED - {result.get('error', 'Unknown error')}")

        # Model comparison
        if len(comparison_data) > 1:
            self.logger.info(f"\nMODEL COMPARISON:")
            self.logger.info("-" * 50)

            # Find best model for each metric
            metrics_to_compare = ['semantic_sim',
                                  'rouge1', 'rouge2', 'rougeL', 'bert_f1']

            for metric in metrics_to_compare:
                if all(metric in data for data in comparison_data):
                    best_model = max(comparison_data, key=lambda x: x[metric])
                    self.logger.info(
                        f"  Best {metric}: {best_model['model']} ({best_model[metric]:.4f})")

        self.logger.info("\n" + "="*80)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("="*80)

    def save_final_results(self):
        """Save the complete results to a JSON file."""
        end_time = datetime.now()

        final_results = {
            'pipeline_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_duration_seconds': (end_time - self.start_time).total_seconds(),
                'config': self.config,
                'log_file': str(self.log_file)
            },
            'results': self.results
        }

        results_file = f"evaluation_results/pipeline_results_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\nFinal results saved to: {results_file}")


def main():
    """Main function to run the evaluation pipeline."""

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run summarization evaluation pipeline')
    parser.add_argument('--approach', choices=['mapreduce', 'iterative', 'truncated', 'mapreduce_critique', 'mapreduce_hierarchical'],
                        default='mapreduce', help='Summarization approach to use')
    parser.add_argument('--models', nargs='+',
                        default=['llama3.2:3b',
                                 'gemma3:4b-it-qat', 'qwen3:8b', 'phi4:14b'],
                        help='List of models to evaluate')
    parser.add_argument('--max-samples', type=int,
                        help='Limit number of documents to process')
    parser.add_argument('--tree-json', default='data_1/document_tree.json',
                        help='Path to tree JSON file for hierarchical approach')
    parser.add_argument('--max-depth', type=int, default=1,
                        help='Maximum depth for hierarchical collapse')

    args = parser.parse_args()

    # Base configuration
    base_config = {
        'approach': args.approach,
        'models': args.models,
        'ollama_url': 'http://localhost:11434',
        'max_new_tokens': 1024,
        'docs_dir': 'data_1/doc',
        'summary_dir': 'data_1/summary',
        'generated_summaries_dir': 'data_1/generated_summaries',
        'evaluation': {
            'embedding_model': 'all-MiniLM-L6-v2',
            'include_llm_eval': True,   # Enable DeepEval LLM-based evaluation
            # Use OpenRouter for LLM evaluation (more reliable)
            'use_openrouter': True,
            'llm_model': 'openai/gpt-4o-mini',  # Model for DeepEval
            'max_samples': args.max_samples
        }
    }

    # Approach-specific configurations
    if args.approach == 'mapreduce':
        approach_config = {
            'chunk_size': 12000,
            'chunk_overlap': 200,
            'token_max': 10000,
        }
    elif args.approach == 'iterative':
        approach_config = {
            'iterative_chunk_size': 12000,
            'iterative_chunk_overlap': 200,
        }
    elif args.approach == 'truncated':
        approach_config = {
            'max_context': 16384,  # Model context window
        }
    elif args.approach == 'mapreduce_critique':
        approach_config = {
            'chunk_size': 12000,
            'chunk_overlap': 200,
            'token_max': 10000,
            'max_critique_iterations': 2,  # Max iterations for critique per collapse
            'max_new_tokens': 2048,  # Higher limit for critique approach
        }
    elif args.approach == 'mapreduce_hierarchical':
        approach_config = {
            'chunk_size': 12000,
            'chunk_overlap': 200,
            'max_depth': args.max_depth,  # Maximum depth for hierarchical collapse
            'tree_json_path': args.tree_json,  # Path to tree structure
        }
    else:
        raise ValueError(f"Unknown approach: {args.approach}")

    # Merge configurations
    config = {**base_config, **approach_config}

    print(
        f"Running {args.approach.upper()} approach with models: {args.models}")
    if args.max_samples:
        print(f"Limited to {args.max_samples} documents")
    if args.approach == 'mapreduce_hierarchical':
        print(f"Using tree structure from: {args.tree_json}")
        print(f"Maximum collapse depth: {args.max_depth}")

    # Create and run pipeline
    runner = PipelineRunner(config)
    asyncio.run(runner.run_full_pipeline())


if __name__ == "__main__":
    main()
