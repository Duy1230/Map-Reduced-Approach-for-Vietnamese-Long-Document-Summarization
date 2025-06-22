import streamlit as st
import random
import asyncio
from pathlib import Path
from typing import Dict, Tuple

# External libs
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Our existing summarisation helpers
from runners.run_summarization_ollama import generate_summary_ollama
from runners.run_summarization_ollama_mapreduce import (
    create_map_reduce_graph, summarize_document_mapreduce
)
from runners.run_summarization_ollama_iterative import (
    create_iterative_refinement_graph, summarize_document_iterative
)
from runners.run_summarization_ollama_mapreduce_critique import (
    create_map_reduce_graph_with_critique, summarize_document_mapreduce_with_critique, create_critique_refine_graph
)
from runners.run_summarization_ollama_mapreduce_hierarchical import (
    hierarchical_summarize_document, OllamaLLM
)


OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "gemma3:4b-it-qat"
DOCS_DIR = Path("data_1/doc")
REF_DIR = Path("data_1/summary")
TREE_JSON_PATH = Path("data_1/document_tree.json")

# Approach identifiers consistent with pipeline
APPROACHES = [
    "truncated",
    "mapreduce",
    "mapreduce_critique",
    "iterative",
    "mapreduce_hierarchical",
]


def load_random_document() -> Tuple[str, str]:
    files = list(DOCS_DIR.glob("*.txt"))
    if not files:
        st.error("data_1/doc directory is empty.")
        st.stop()
    f = random.choice(files)
    return f.stem, f.read_text(encoding="utf-8")


def get_reference_summary(name_stem: str) -> str | None:
    ref_path = REF_DIR / f"{name_stem}.txt"
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8")
    return None


def compute_metrics(reference: str, generated: str) -> Dict[str, float]:
    """Return ROUGE-1/2/L F1 and BERT F1."""
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    rouge1 = scores['rouge1'].fmeasure
    rouge2 = scores['rouge2'].fmeasure
    rougel = scores['rougeL'].fmeasure

    # BERTScore (single pair)
    P, R, F1 = bert_score([generated], [reference],
                          lang="vi", rescale_with_baseline=True)
    bert_f1 = F1.mean().item()
    return {
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougel,
        "BERT F1": bert_f1,
    }


@st.cache_resource(show_spinner=False)
def get_llm():
    return OllamaLLM(ollama_url=OLLAMA_URL, model_name=MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b")


async def _summarise_async(approach: str, text: str, name_stem: str):
    llm = get_llm()
    tokenizer = get_tokenizer()

    if approach == "truncated":
        max_context = 16384
        max_new_tokens = 2048
        max_input_tokens = max_context - max_new_tokens
        summary = generate_summary_ollama(
            text, tokenizer, max_input_tokens, max_new_tokens, OLLAMA_URL, MODEL_NAME
        )
        return summary

    elif approach == "mapreduce":
        text_splitter = None  # constructed within summarise_document_mapreduce
        graph = create_map_reduce_graph(llm, token_max=10000)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=12000,
            chunk_overlap=200,
            length_function=llm.get_num_tokens,
            separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""],
        )
        summary = await summarize_document_mapreduce(text, graph, splitter)
        return summary

    elif approach == "mapreduce_critique":
        map_app = create_map_reduce_graph_with_critique(
            llm, token_max=10000, max_critique_iterations=2)
        critique_app = create_critique_refine_graph(llm)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=12000,
            chunk_overlap=200,
            length_function=llm.get_num_tokens,
            separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""],
        )
        summary = await summarize_document_mapreduce_with_critique(text, map_app, critique_app, splitter)
        return summary

    elif approach == "iterative":
        graph = create_iterative_refinement_graph(llm)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=12000,
            chunk_overlap=200,
            length_function=llm.get_num_tokens,
            separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""],
        )
        summary = await summarize_document_iterative(text, graph, splitter)
        return summary

    elif approach == "mapreduce_hierarchical":
        # Load tree and locate document node
        if not TREE_JSON_PATH.exists():
            return "(No tree file found)"
        import json
        import copy
        tree = json.loads(TREE_JSON_PATH.read_text(encoding="utf-8"))
        document_node = None
        for doc in tree.get("children", []):
            if doc.get("type") == "Document" and doc.get("text") == name_stem:
                document_node = copy.deepcopy(doc)
                break
        if document_node is None:
            return "(Document not found in tree)"
        summary = await hierarchical_summarize_document(
            document_node, max_depth=2, llm=llm, chunk_size=12000, chunk_overlap=200
        )
        return summary

    else:
        raise ValueError("Unknown approach")


def summarise(approach: str, text: str, name_stem: str):
    """Synchronous wrapper for Streamlit."""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, create new thread event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, _summarise_async(approach, text, name_stem))
                return future.result()
        else:
            return loop.run_until_complete(_summarise_async(approach, text, name_stem))
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(_summarise_async(approach, text, name_stem))


st.title("Multi-approach Document Summarisation Demo")

st.sidebar.header("Document source")
source_mode = st.sidebar.radio(
    "Select source", ["Random from data_1/doc", "Upload text file"])

# Initialize session state to persist document selection
if 'doc_selected' not in st.session_state:
    st.session_state.doc_selected = False
if 'doc_name' not in st.session_state:
    st.session_state.doc_name = ""
if 'doc_text' not in st.session_state:
    st.session_state.doc_text = ""

if source_mode == "Random from data_1/doc":
    if st.sidebar.button("Pick random document"):
        doc_name, doc_text = load_random_document()
        st.session_state.doc_name = doc_name
        st.session_state.doc_text = doc_text
        st.session_state.doc_selected = True
        st.sidebar.success(f"Selected: {doc_name}")

    if not st.session_state.doc_selected:
        st.info("Click 'Pick random document' to get started!")
        st.stop()
    else:
        doc_name = st.session_state.doc_name
        doc_text = st.session_state.doc_text
else:
    uploaded = st.sidebar.file_uploader("Upload .txt file", type=["txt"])
    if uploaded is None:
        st.info("Upload a text file to get started!")
        st.stop()
    doc_name = uploaded.name.rsplit(".", 1)[0]
    doc_text = uploaded.read().decode("utf-8")

# Display document preview
with st.expander("Show document text (first 1000 chars)"):
    st.write(doc_text[:1000] + (" …" if len(doc_text) > 1000 else ""))

# Load reference summary if available
reference = get_reference_summary(doc_name)
if reference:
    with st.expander("Ground-truth summary"):
        st.write(reference)
else:
    st.warning("No reference summary available. Quality metrics will be skipped.")

if st.button("Run summarisation (5 approaches)"):
    results: Dict[str, Dict] = {}

    # Create containers for dynamic updates
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    results_placeholder = st.empty()

    total = len(APPROACHES)

    for i, approach in enumerate(APPROACHES):
        # Update progress
        progress_value = i / total
        progress_bar.progress(progress_value)
        status_text.text(f"Running {approach}... ({i+1}/{total})")

        try:
            # Run summarization
            summary = summarise(approach, doc_text, doc_name)

            # Compute metrics if reference available
            if reference:
                metrics = compute_metrics(reference, summary)
            else:
                metrics = {}

            results[approach] = {"summary": summary, "metrics": metrics}

        except Exception as e:
            st.error(f"Error running {approach}: {str(e)}")
            results[approach] = {"summary": f"Error: {str(e)}", "metrics": {}}

    # Final progress update
    progress_bar.progress(1.0)
    status_text.text("Completed!")

    # Clear status after brief pause
    import time
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()

    st.success("Summarisation completed!")

    # Display results in tabs
    tabs = st.tabs([ap.replace("_", " ").title() for ap in APPROACHES])
    for tab, approach in zip(tabs, APPROACHES):
        with tab:
            st.subheader(f"{approach.replace('_', ' ').title()} Summary")
            st.write(results[approach]["summary"])
            if reference and results[approach]["metrics"]:
                st.markdown("**Metrics (vs ground truth):**")
                # Display metrics in a nice table format
                metrics_data = []
                for k, v in results[approach]["metrics"].items():
                    metrics_data.append({"Metric": k, "Score": f"{v:.4f}"})
                st.table(metrics_data)
