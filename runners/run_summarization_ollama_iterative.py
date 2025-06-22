import os
import requests
import asyncio
from typing import List, Literal, TypedDict
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
import re


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

        # Get the raw response and clean thinking tokens
        raw_response = resp.json()["response"]
        cleaned_response = clean_thinking_tokens(raw_response)

        return cleaned_response

    async def _acall(self, prompt: str, stop=None, run_manager=None, **kwargs):
        return self._call(prompt, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def get_num_tokens(self, text: str) -> int:
        # Simple approximation - could be improved with proper tokenizer
        return len(text.split())


# Iterative Refinement State Definition
class IterativeState(TypedDict):
    contents: List[str]
    index: int
    summary: str


def create_iterative_refinement_graph(llm):
    """Create the Iterative Refinement graph for summarization"""

    # Vietnamese prompts for initial summary
    initial_summarize_prompt = ChatPromptTemplate.from_template(
        """Bạn là một chuyên gia phân tích và tóm tắt thông tin.
Nhiệm vụ của bạn là đọc phần đầu tiên của một tài liệu dài và tạo ra một bản tóm tắt **nền tảng**.

Bản tóm tắt này phải nắm bắt được những ý chính, bối cảnh và các thông tin quan trọng nhất làm cơ sở cho việc xây dựng một bản tóm tắt toàn diện sau này. Hãy tập trung vào việc xác định các yếu tố cốt lõi (Ai, Cái gì, Khi nào, Ở đâu, Tại sao) được giới thiệu trong đoạn văn này.

Văn bản cần tóm tắt:
---
{context}
---

Bản tóm tắt nền tảng:
"""
    )
    # Vietnamese prompts for refinement
    refine_template = """
Bạn là một biên tập viên xuất sắc, chuyên tổng hợp và tinh chỉnh thông tin từ nhiều nguồn.
Nhiệm vụ của bạn là cập nhật và mở rộng một bản tóm tắt đã có với những thông tin mới.

Bản tóm tắt hiện có (tóm tắt các phần trước):
---
{existing_answer}
---

Thông tin mới cần tích hợp (từ phần văn bản tiếp theo):
---
{context}
---

Dựa vào thông tin mới, hãy **viết lại hoàn toàn** bản tóm tắt để tạo ra một phiên bản mới, mạch lạc và toàn diện hơn.

**Yêu cầu quan trọng:**
1.  **Tích hợp, không nối thêm:** Đừng chỉ viết thêm thông tin mới vào cuối. Hãy khéo léo lồng ghép các chi tiết mới vào bản tóm tắt hiện có, sắp xếp lại các câu và ý tưởng để tạo ra một dòng chảy tự nhiên.
2.  **Bảo toàn thông tin cốt lõi:** Đảm bảo rằng những điểm chính và bối cảnh quan trọng từ "Bản tóm tắt hiện có" không bị mất đi hoặc giảm nhẹ tầm quan trọng, trừ khi thông tin mới làm rõ hoặc thay đổi chúng một cách trực tiếp.
3.  **Tổng hợp và cân bằng:** Bản tóm tắt cuối cùng phải phản ánh một cách cân bằng toàn bộ nội dung đã biết cho đến nay, không thiên vị cho thông tin mới nhất.

Hãy viết bản tóm tắt tổng hợp cuối cùng bằng câu văn hoàn chỉnh, liền mạch thành một đoạn văn bằng tiếng Việt.

Bản tóm tắt tổng hợp cuối cùng:
"""

    refine_prompt = ChatPromptTemplate([("human", refine_template)])

    # Create chains
    initial_summary_chain = initial_summarize_prompt | llm | StrOutputParser()
    refine_summary_chain = refine_prompt | llm | StrOutputParser()

    # Generate initial summary from first chunk
    async def generate_initial_summary(state: IterativeState, config: RunnableConfig = None):
        summary = await initial_summary_chain.ainvoke(
            {"context": state["contents"][0]},
            config or {},
        )
        return {"summary": summary, "index": 1}

    # Refine summary with next chunk
    async def refine_summary(state: IterativeState, config: RunnableConfig = None):
        content = state["contents"][state["index"]]
        summary = await refine_summary_chain.ainvoke(
            {"existing_answer": state["summary"], "context": content},
            config or {},
        )

        return {"summary": summary, "index": state["index"] + 1}

    # Determine if we should continue refining or end
    def should_refine(state: IterativeState) -> Literal["refine_summary", END]:
        if state["index"] >= len(state["contents"]):
            return END
        else:
            return "refine_summary"

    # Build the graph
    graph = StateGraph(IterativeState)
    graph.add_node("generate_initial_summary", generate_initial_summary)
    graph.add_node("refine_summary", refine_summary)

    graph.add_edge(START, "generate_initial_summary")
    graph.add_conditional_edges("generate_initial_summary", should_refine)
    graph.add_conditional_edges("refine_summary", should_refine)

    return graph.compile()


async def summarize_document_iterative(doc_text: str, app, text_splitter):
    """Summarize a document using Iterative Refinement approach"""

    # Create document and split it
    doc = Document(page_content=doc_text)
    split_docs = text_splitter.split_documents([doc])

    print(
        f"Split document into {len(split_docs)} chunks for iterative refinement")

    # Prepare contents for iterative refinement
    contents = [doc.page_content for doc in split_docs]

    # Run the Iterative Refinement graph
    result = await app.ainvoke({
        "contents": contents,
        "index": 0,
        "summary": ""
    })

    return result["summary"]


async def main():
    OLLAMA_URL = "http://localhost:11434"
    MODEL_NAME = "llama3.2:3b"  # You can change this to other models

    # Initialize LLM
    llm = OllamaLLM(ollama_url=OLLAMA_URL,
                    model_name=MODEL_NAME, max_new_tokens=2048)

    # Initialize tokenizer for accurate token-based splitting
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b")

    # Create custom length function using the actual tokenizer
    def length_function(text: str) -> int:
        return len(tokenizer.encode(text))

    # Use RecursiveCharacterTextSplitter with custom length function
    # For iterative refinement, we want larger chunks to maintain context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,  # Larger chunks for iterative refinement
        chunk_overlap=400,  # Higher overlap to maintain continuity
        length_function=length_function,
        separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
    )

    # Create Iterative Refinement graph
    app = create_iterative_refinement_graph(llm)

    docs_dir = "data/doc"
    summary_dir = "data/summary"
    generated_summaries_dir = "data/generated_summaries_iterative"

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

        # Get document length for reporting
        doc_tokens = tokenizer.encode(doc_text)
        print(f"Processing {fname}... ({len(doc_tokens)} tokens)")

        # Generate summary using Iterative Refinement
        summary = await summarize_document_iterative(doc_text, app, text_splitter)

        # Additional cleaning to ensure no thinking tokens remain
        summary = clean_thinking_tokens(summary)

        # Save generated summary immediately
        with open(gen_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Saved generated summary to {gen_path}")

        generated_summaries.append(summary)
        references.append(ref)

    if not generated_summaries or not references:
        print("No summaries generated or no references found for evaluation")
        return

    # Evaluate with ROUGE
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougel = [], [], []
    for gen, ref in zip(generated_summaries, references):
        scores = scorer.score(ref, gen)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougel.append(scores['rougeL'].fmeasure)

    print("\n=== EVALUATION RESULTS ===")
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
    asyncio.run(main())
