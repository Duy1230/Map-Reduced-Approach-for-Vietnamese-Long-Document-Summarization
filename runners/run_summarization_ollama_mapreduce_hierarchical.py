from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Literal

import requests
from langgraph.graph import END, START, StateGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.llms import LLM


def clean_thinking_tokens(text: str) -> str:
    """Remove typical <think> … </think> blocks that Ollama models might emit."""
    if not text:
        return text

    import re

    patterns = [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"<thought>.*?</thought>",
        r"<reasoning>.*?</reasoning>",
        r"<analysis>.*?</analysis>",
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    # normalise whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class OllamaLLM(LLM):
    """Minimal LLM wrapper around the Ollama HTTP API with optional token cleaning."""

    ollama_url: str = "http://localhost:11434"
    model_name: str = "llama3.2:3b"
    max_new_tokens: int = 2048
    _clean_thinking: bool = True

    # NOTE: pydantic will treat all __init__ args as field overrides
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # type: ignore[override]
    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": self.max_new_tokens},
            "think": False,
        }
        resp = requests.post(
            f"{self.ollama_url}/api/generate", json=payload, timeout=600)
        resp.raise_for_status()
        output = resp.json()["response"]
        return clean_thinking_tokens(output) if self._clean_thinking else output

    # type: ignore[override]
    async def _acall(self, prompt: str, stop=None, run_manager=None, **kwargs):
        return self._call(prompt, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "ollama"

    # crude token estimate
    def get_num_tokens(self, text: str) -> int:
        return len(text.split())


MAP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Bạn là một chuyên gia tóm tắt nội dung. Hãy tóm tắt những thông tin quan trọng từ đoạn văn bản sau bằng tiếng Việt.\n"
                "Lưu ý bao gồm đầy đủ các chi tiết quan trọng như sự kiện hay nhân vật, các chủ đề chính. Không bỏ sót thông tin quan trọng. Nên tóm tắt theo từng chương nếu có."
                "<content>\n"
                "{content}\n\n"
                "</content>\n\n"
                # "Yêu cầu:\n"
                # "1. Mở đầu bằng 1–2 câu nêu chủ đề chính.\n"
                # "2. Trình bày ngắn gọn các luận điểm hoặc chi tiết then chốt.\n"
                # "3. Không sử dụng dấu đầu dòng; hãy viết thành các câu hoàn chỉnh theo đoạn văn.\n"
                "Chỉ viết nội dung tóm tắt. Không giải thích, không xin lỗi, không nói về quy trình.\n"
                # "5. Nên tóm tắt theo từng chương nếu có.\n"
                "Tóm tắt:"
            ),
        )
    ]
)

REDUCE_TEMPLATE = (
    "Sau đây là một tập hợp các bản tóm tắt:\n<docs>\n{docs}\n</docs>\n\n"
    "Hãy tổng hợp và chắt lọc chúng thành một bản tóm tắt cuối cùng bằng **tiếng Việt**\n"
    "Lưu ý bao gồm đầy đủ các chi tiết quan trọng như sự kiện hay nhân vật, các chủ đề chính. Không bỏ sót thông tin quan trọng."
    "Chỉ viết nội dung tóm tắt. Không giải thích, không xin lỗi, không nói về quy trình."
    "Không sử dụng dấu đầu dòng; hãy viết thành các câu hoàn chỉnh theo đoạn văn."
    # "Nên tóm tắt theo từng chương nếu có."
    "Tóm tắt mới:"
)

REDUCE_PROMPT = ChatPromptTemplate([("human", REDUCE_TEMPLATE)])


class MapReduceState(TypedDict):
    contents: List[str]
    summaries: List[str]
    final_summary: str
    index: int  # internal pointer for iteration


def create_simple_mapreduce_graph(llm: LLM):
    """Create a lightweight map-reduce graph for summarization."""

    initial_chain = MAP_PROMPT | llm | StrOutputParser()
    reduce_chain = REDUCE_PROMPT | llm | StrOutputParser()

    # generate summary for each chunk (MAP)
    async def map_chunk(state: MapReduceState):
        chunk = state["contents"][state["index"]]
        summary = await initial_chain.ainvoke({"content": chunk})
        return {"summaries": state.get("summaries", []) + [summary], "index": state["index"] + 1}

    # if more chunks remain continue mapping else proceed to reduce
    def should_continue(state: MapReduceState) -> Literal["map_chunk", "reduce_chunks", END]:
        if state["index"] < len(state["contents"]):
            return "map_chunk"
        return "reduce_chunks"

    async def reduce_chunks(state: MapReduceState):
        joined = "\n\n".join(state["summaries"])
        summary = await reduce_chain.ainvoke({"docs": joined})
        return {"final_summary": summary}

    g = StateGraph(MapReduceState)
    g.add_node("map_chunk", map_chunk)
    g.add_node("reduce_chunks", reduce_chunks)
    g.add_edge(START, "map_chunk")
    g.add_conditional_edges("map_chunk", should_continue)
    g.add_edge("reduce_chunks", END)
    return g.compile()


_MAP_REDUCE_APP_CACHE: dict[int, any] = {}


def _get_mapreduce_app(llm: LLM):
    """Return cached simple map-reduce graph for *llm*, compiling once if needed."""
    key = id(llm)
    if key not in _MAP_REDUCE_APP_CACHE:
        _MAP_REDUCE_APP_CACHE[key] = create_simple_mapreduce_graph(llm)
    return _MAP_REDUCE_APP_CACHE[key]


async def summarize_text_mapreduce(
    text: str,
    llm: LLM,
    *,
    chunk_size: int = 12000,
    chunk_overlap: int = 200,
    max_context: int = 16384,
) -> str:
    """Split *text* into chunks (≤75 % context) and summarise with a cached map-reduce graph."""

    max_safe_tokens = int(max_context * 0.75)
    effective_chunk_size = min(chunk_size, max_safe_tokens)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=llm.get_num_tokens,  # token-aware splitting
        separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""],
    )

    docs = splitter.split_text(text)

    app = _get_mapreduce_app(llm)

    result = await app.ainvoke({
        "contents": docs,
        "summaries": [],
        "index": 0,
        "final_summary": "",
    })

    return result["final_summary"]


def depth_first_traverse(node: Dict, callback, depth: int = 0, parent: Optional[Dict] = None):
    """Generic DFS traversal executing *callback(node, depth, parent)*."""
    callback(node, depth, parent)
    for child in node.get("children", []):
        depth_first_traverse(child, callback, depth + 1, node)


def collect_nodes_at_depth(root: Dict, target_depth: int) -> List[Dict]:
    nodes: List[Dict] = []

    def _cb(node, depth, _parent):
        if depth == target_depth and node.get("type") != "Paragraph":
            nodes.append(node)

    depth_first_traverse(root, _cb)
    return nodes


def extract_descendant_paragraph_text(node: Dict) -> str:
    """Concatenate text of all descendant Paragraph nodes of *node*."""
    texts: List[str] = []

    def _cb(n, _d, _p):
        if n.get("type") == "Paragraph":
            texts.append(n.get("text", ""))

    depth_first_traverse(node, _cb)
    return "\n\n".join(texts)


def replace_node_with_paragraph(node: Dict, summary_text: str):
    """Mutate *node* in-place turning it into a Paragraph leaf containing *summary_text*."""
    # Remove children
    node.pop("children", None)
    # Replace core fields
    node.clear()
    node["type"] = "Paragraph"
    node["text"] = summary_text


async def collapse_level(root: Dict, depth_level: int, llm: LLM, chunk_size: int, chunk_overlap: int):
    """Summarise all non-leaf nodes at *depth_level* and replace them with Paragraphs."""
    targets = collect_nodes_at_depth(root, depth_level)
    if not targets:
        return 0

    for target in targets:
        # Preserve the header/section title if available
        header_title = target.get("text", "").strip()

        # Gather the body text from descendant paragraphs
        body_text = extract_descendant_paragraph_text(target)

        if not body_text.strip():
            # No text -> replace node with an empty paragraph (retain title to avoid losing structure)
            replace_node_with_paragraph(target, header_title)
            continue

        # Combine title and body so the LLM understands the section context
        input_text = f"{header_title}\n\n{body_text}" if header_title else body_text

        summary_core = await summarize_text_mapreduce(
            input_text,
            llm,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Prefix the summary with the original header title so downstream levels retain structure
        summary_with_title = f"{header_title}:\n{summary_core}" if header_title else summary_core

        replace_node_with_paragraph(target, summary_with_title)
    return len(targets)


async def hierarchical_summarize_document(document_node: Dict, *, max_depth: int, llm: LLM, chunk_size: int = 12000, chunk_overlap: int = 200) -> str:
    """Collapse a *Document* node bottom-up and return final summary string."""
    # Determine actual max depth if not provided (scan tree)
    def _get_depth(n: Dict, depth: int = 0):
        if not n.get("children"):
            return depth
        return max(_get_depth(c, depth + 1) for c in n["children"])

    actual_max_depth = _get_depth(document_node)
    target_max_depth = min(max_depth, actual_max_depth)

    # Collapse from deepest -> 1 (headers directly under Document)
    for d in range(target_max_depth, 0, -1):
        await collapse_level(document_node, d, llm, chunk_size, chunk_overlap)

    # Now document_node children should all be Paragraphs.  Concatenate and summarise once more.
    final_text = extract_descendant_paragraph_text(document_node)
    final_summary = await summarize_text_mapreduce(final_text, llm, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    review_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "Bạn là một biên tập viên chuyên nghiệp.\n"
                "Dưới đây là bản tóm tắt của một tài liệu:\n"
                "<summary>\n"
                "{summary}"
                "</summary>\n"

                "Hãy rà soát để sửa lỗi ngữ pháp và đảm bảo văn phong mạch lạc, rõ ràng. Không bỏ sót thông tin quan trọng.\n"
                "không cần giải thích, không cần xin lỗi, không cần nói về quy trình.\n"
                "Tóm tắt mới:\n"
            ),
        )
    ])

    polished_summary = await llm.ainvoke(review_prompt.invoke({"summary": final_summary}).messages[0].content)

    return polished_summary


async def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Hierarchical Map-Reduce summarization runner")
    parser.add_argument("--tree-json", required=True,
                        help="Path to JSON file containing the document tree")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to store generated summaries")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="Maximum depth at which to start collapsing (Document=0)")
    parser.add_argument("--model", default="llama3.2:3b",
                        help="Ollama model name")
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--chunk-size", type=int, default=12000,
                        help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int,
                        default=200, help="Chunk overlap for text splitting")
    args = parser.parse_args()

    # Load tree
    with open(args.tree_json, "r", encoding="utf-8") as f:
        tree = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    llm = OllamaLLM(ollama_url=args.ollama_url, model_name=args.model)

    # Expect tree root to have children Documents
    docs = [n for n in tree.get("children", []) if n.get("type") == "Document"]
    if not docs:
        print("No Document nodes found in tree.", file=sys.stderr)
        sys.exit(1)

    for doc in docs:
        doc_name = doc.get("text", "unnamed_doc")
        print(f"Summarising '{doc_name}' ...")
        summary = await hierarchical_summarize_document(
            doc,
            max_depth=args.max_depth,
            llm=llm,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        out_path = Path(args.output_dir) / f"{doc_name}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Saved summary → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
