import os
import requests
import operator
import asyncio
from typing import Annotated, List, Literal, TypedDict
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
import re
from langsmith import traceable
import dotenv
dotenv.load_dotenv()


def clean_thinking_tokens(text: str) -> str:
    """Remove thinking tokens from model output."""
    if not text:
        return text

    patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<thought>.*?</thought>',
        r'<reasoning>.*?</reasoning>',
        r'<analysis>.*?</analysis>'
    ]

    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text,
                              flags=re.DOTALL | re.IGNORECASE)

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
            },
            "think": False
        }

        resp = requests.post(f"{self.ollama_url}/api/generate", json=payload)
        resp.raise_for_status()

        raw_response = resp.json()["response"]
        cleaned_response = clean_thinking_tokens(raw_response)
        return cleaned_response

    async def _acall(self, prompt: str, stop=None, run_manager=None, **kwargs):
        return self._call(prompt, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())


# Enhanced State Definitions for Critique-Integrated MapReduce
class OverallState(TypedDict):
    contents: List[str]
    original_chunks: List[str]  # Keep original chunks for critique
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str
    critique_iterations: int


class SummaryState(TypedDict):
    content: str


class CritiqueCollapseState(TypedDict):
    docs_to_collapse: List[Document]
    original_chunks: List[str]  # Corresponding original chunks
    collapsed_result: str


def create_map_reduce_graph_with_critique(llm, token_max: int = 1000, max_critique_iterations: int = 2):
    """Create enhanced Map-Reduce graph with integrated critique during collapse phases"""

    # BETTER ABSTRACTIVE PROMPTS

    # Initial chunk summarization - simple and direct
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", """Hãy tóm tắt những thông tin quan trọng từ đoạn văn bản sau bằng tiếng Việt.
        Lưu ý bao gồm đầy đủ các chi tiết quan trọng như sự kiện hay nhân vật, các chủ đề chính. Không bỏ sót thông tin quan trọng. Nên tóm tắt theo từng chương nếu có.

Chỉ viết nội dung tóm tắt. Không giải thích, không xin lỗi, không nói về quy trình.

Văn bản:
<content>
{content}
</content>

Tóm tắt:""")
    ])

    # Collapse/Reduce prompt - with section structure awareness
    reduce_template = """
Hãy kết hợp các bản tóm tắt được đánh dấu theo phần sau thành MỘT bản tóm tắt duy nhất bằng tiếng Việt.

Các bản tóm tắt theo phần:
<summary>
{docs}
</summary>

Yêu cầu tổng hợp: Tổng hợp các thông tin từ TẤT CẢ các phần theo trình tự logic. Tạo ra một câu chuyện/tóm tắt liền mạch, kết nối các phần với nhau. Bao gồm đầy đủ các chi tiết quan trọng như sự kiện, nhân vật, chủ đề chính. Không bỏ sót thông tin quan trọng từ bất kỳ phần nào. Giữ nguyên trình tự thời gian/logic nếu có.

Chỉ viết nội dung tóm tắt tổng hợp cuối cùng. Không đề cập đến các tag phần, không giải thích quy trình.

Tóm tắt tổng hợp:
"""

    # CRITIQUE PROMPT for collapse phase - simplified
    critique_collapse_template = """
So sánh bản tóm tắt với nội dung tham khảo. Có thông tin quan trọng nào bị thiếu hoặc sai không?
Các thông tin quan trọng bao gồm sự kiện hay nhân vật,các chủ đề chính. Không bỏ sót thông tin quan trọng.

Bản tóm tắt:
<summary>
{summary}
</summary>

Nội dung tham khảo:
<reference_content>
{original_chunks}
</reference_content>

Nếu không có vấn đề thì trả lời: "Không có vấn đề"
Nếu có vấn đề thì chỉ ra vấn đề cụ thể thật chi tiết và rõ ràng. không cần giải thích, không cần xin lỗi, không cần nói về quy trình.
Ví dụ: "Thiếu thông tin về sự kiện X", "Thiếu thông tin về nhân vật Y"
"""

    # REFINE PROMPT for collapse phase - with reference context
    refine_collapse_template = """
Nhiệm vụ: Viết lại bản tóm tắt để khắc phục các vấn đề đã chỉ ra. Sử dụng nội dung tham khảo để bổ sung thông tin bị thiếu.

Bản tóm tắt hiện tại (cần sửa):
<summary>
{current_summary}
</summary>

Vấn đề cần khắc phục:
<critique>
{critique}
</critique>

Nội dung tham khảo (để bổ sung thông tin):
<reference_content>
{reference_content}
</reference_content>

Yêu cầu:
- Khắc phục TẤT CẢ các vấn đề đã chỉ ra trong phần critique
- Bổ sung thông tin bị thiếu từ nội dung tham khảo
- Giữ nguyên thông tin đúng đã có trong bản tóm tắt cũ
- Đảm bảo tóm tắt mới có đầy đủ thông tin và chính xác

Chỉ viết bản tóm tắt đã sửa. Không giải thích, không xin lỗi, không nói về quy trình.

Bản tóm tắt đã sửa:
"""

    reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
    critique_prompt = ChatPromptTemplate(
        [("human", critique_collapse_template)])
    refine_prompt = ChatPromptTemplate([("human", refine_collapse_template)])

    def length_function(documents: List[Document]) -> int:
        return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

    # Generate summary for a single document chunk
    async def generate_summary(state: SummaryState):
        prompt = map_prompt.invoke({"content": state["content"]})
        response = await llm.ainvoke(prompt.messages[0].content)
        return {"summaries": [response]}

    # Map summaries across all documents
    def map_summaries(state: OverallState):
        return [
            Send("generate_summary", {"content": content}) for content in state["contents"]
        ]

    def collect_summaries(state: OverallState):
        return {
            "collapsed_summaries": [Document(page_content=summary) for summary in state["summaries"]]
        }

    # ENHANCED REDUCE WITH CRITIQUE INTEGRATION
    async def _reduce_with_critique(docs_input, original_chunks_input, iteration=0) -> str:
        """Reduce function with integrated critique loop"""

        if isinstance(docs_input, list):
            # Add section tags to clearly separate each summary
            tagged_docs = []
            for i, doc in enumerate(docs_input):
                section_tag = f"[PHẦN {i+1}]"
                tagged_docs.append(f"{section_tag}\n{doc.page_content}")
            docs_text = "\n\n".join(tagged_docs)
        else:
            docs_text = str(docs_input)

        # Initial reduce
        prompt = reduce_prompt.invoke({"docs": docs_text})
        initial_summary = await llm.ainvoke(prompt.messages[0].content)

        # Skip critique on the first iteration or if we've reached max iterations
        if iteration >= max_critique_iterations:
            return initial_summary

        # Critique phase
        original_chunks_text = "\n\n---\n\n".join(original_chunks_input)
        critique_prompt_filled = critique_prompt.invoke({
            "summary": initial_summary,
            "original_chunks": original_chunks_text
        })
        critique_result = await llm.ainvoke(critique_prompt_filled.messages[0].content)

        # Check if critique found issues
        if "không có vấn đề" in critique_result.lower() or "no issues" in critique_result.lower():
            return initial_summary

        # Refine based on critique with reference content
        refine_prompt_filled = refine_prompt.invoke({
            "current_summary": initial_summary,
            "critique": critique_result,
            "reference_content": original_chunks_text  # Include reference content
        })
        refined_summary = await llm.ainvoke(refine_prompt_filled.messages[0].content)

        return refined_summary

    # Enhanced collapse summaries with critique
    async def collapse_summaries(state: OverallState):
        doc_lists = split_list_of_docs(
            state["collapsed_summaries"], length_function, token_max
        )

        results = []
        current_chunk_index = 0

        for doc_list in doc_lists:
            # Get corresponding original chunks for this doc_list
            chunks_for_this_group = state["original_chunks"][current_chunk_index:current_chunk_index + len(
                doc_list)]
            current_chunk_index += len(doc_list)

            # Reduce with critique
            result = await _reduce_with_critique(
                doc_list,
                chunks_for_this_group,
                state.get("critique_iterations", 0)
            )

            results.append(Document(page_content=str(result)))

        return {
            "collapsed_summaries": results,
            "critique_iterations": state.get("critique_iterations", 0) + 1
        }

    # Determine if we should collapse summaries or generate final summary
    def should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
        num_tokens = length_function(state["collapsed_summaries"])
        if num_tokens > token_max:
            return "collapse_summaries"
        else:
            return "generate_final_summary"

        # Generate the final summary with critique using intermediate summaries
    async def generate_final_summary(state: OverallState):
        # BETTER APPROACH: Use the already-critiqued intermediate summaries
        # These represent full coverage of the document and are much shorter
        intermediate_summaries = [
            doc.page_content for doc in state["collapsed_summaries"]]

        # Check if we can fit all intermediate summaries in context
        total_tokens = sum(llm.get_num_tokens(summary)
                           for summary in intermediate_summaries)

        if total_tokens <= token_max // 2:  # Use half token limit to leave room for critique
            # We can use ALL intermediate summaries - this gives full coverage!
            critique_context = intermediate_summaries
        else:
            # RECURSIVE COLLAPSE: Further reduce intermediate summaries while preserving all information
            print(
                f"   Intermediate summaries too long ({total_tokens} tokens), applying recursive collapse...")

            # Convert back to Document objects for recursive collapse
            intermediate_docs = [Document(page_content=summary)
                                 for summary in intermediate_summaries]

            # Apply another round of collapse with critique to reduce size
            # Group intermediate summaries if there are many of them
            doc_lists = split_list_of_docs(
                intermediate_docs, length_function, token_max // 2)

            reduced_summaries = []
            for doc_list in doc_lists:
                # Use the intermediate summaries themselves as the reference for critique
                current_summaries = [doc.page_content for doc in doc_list]

                reduced_result = await _reduce_with_critique(
                    doc_list,
                    current_summaries,  # Use the summaries being collapsed as reference
                    state.get("critique_iterations", 0)
                )
                reduced_summaries.append(str(reduced_result))

            critique_context = reduced_summaries
            print(
                f"   Reduced to {len(critique_context)} summaries ({sum(llm.get_num_tokens(s) for s in critique_context)} tokens)")

        response = await _reduce_with_critique(
            state["collapsed_summaries"],
            critique_context,  # Use intermediate summaries, not original chunks
            state.get("critique_iterations", 0)
        )

        if isinstance(response, Document):
            final_summary = response.page_content
        else:
            final_summary = str(response)
        return {"final_summary": final_summary}

    # Construct the graph
    graph = StateGraph(OverallState)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)

    # Edges
    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)

    return graph.compile()


# Keep the old functions for backward compatibility but mark as deprecated
def create_map_reduce_graph(llm, token_max: int = 1000):
    """DEPRECATED: Use create_map_reduce_graph_with_critique instead"""
    return create_map_reduce_graph_with_critique(llm, token_max, max_critique_iterations=0)


def create_critique_refine_graph(llm, max_iterations: int = 3):
    """DEPRECATED: Critique is now integrated into the map-reduce process"""
    # Return a dummy graph that just passes through
    from langgraph.graph import StateGraph

    class DummyState(TypedDict):
        pass

    graph = StateGraph(DummyState)

    async def dummy_node(state):
        return state

    graph.add_node("dummy", dummy_node)
    graph.add_edge(START, "dummy")
    graph.add_edge("dummy", END)

    return graph.compile()


@traceable
async def summarize_document_mapreduce_with_critique(
    doc_text: str,
    mapreduce_app,
    critique_app,  # Ignored in new implementation
    text_splitter,
    max_critique_iterations: int = 2
):
    """
    Enhanced summarization with integrated critique during collapse phases
    """

    # Create document and split it
    doc = Document(page_content=doc_text)
    split_docs = text_splitter.split_documents([doc])

    print(f"Split document into {len(split_docs)} chunks")
    print(
        f"Using integrated critique with max {max_critique_iterations} iterations per collapse")

    # Prepare both content and original chunks for the enhanced process
    contents = [doc.page_content for doc in split_docs]
    original_chunks = contents.copy()  # Keep original chunks for critique

    # Run the enhanced Map-Reduce with integrated critique
    result = None
    async for step in mapreduce_app.astream(
        {
            "contents": contents,
            "original_chunks": original_chunks,
            "summaries": [],
            "collapsed_summaries": [],
            "final_summary": "",
            "critique_iterations": 0
        },
        {"recursion_limit": 15},  # Increased for critique loops
    ):
        if "generate_final_summary" in step:
            result = step["generate_final_summary"]["final_summary"]

    return result


async def main():
    OLLAMA_URL = "http://localhost:11434"
    MODEL_NAME = "llama3.2:3b"

    # Initialize LLM
    llm = OllamaLLM(ollama_url=OLLAMA_URL,
                    model_name=MODEL_NAME, max_new_tokens=2048)

    # Initialize text splitter
    text_splitter = CharacterTextSplitter(
        chunk_size=2000,  # Smaller for demo
        chunk_overlap=100,
        separator="\n\n"
    )

    # Create the enhanced graph with integrated critique
    mapreduce_app = create_map_reduce_graph_with_critique(
        llm,
        token_max=1500,
        max_critique_iterations=2
    )

    # Dummy critique app for compatibility
    critique_app = create_critique_refine_graph(llm)

    # Tokenizer for evaluation
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b")

    docs_dir = "data/doc"
    summary_dir = "data/summary"
    generated_summaries_dir = "data/generated_summaries_mapreduce_critique"

    os.makedirs(generated_summaries_dir, exist_ok=True)

    generated_summaries = []
    references = []

    print("🚀 Starting Enhanced Map-Reduce with Integrated Critique")
    print("=" * 60)

    for fname in sorted(os.listdir(docs_dir)):
        doc_path = os.path.join(docs_dir, fname)
        ref_path = os.path.join(summary_dir, fname)
        gen_path = os.path.join(generated_summaries_dir, fname)

        if os.path.isfile(gen_path):
            print(f"📄 {fname}: Already exists, loading...")
            with open(gen_path, 'r', encoding='utf-8') as f:
                generated_summaries.append(f.read())
            if os.path.isfile(ref_path):
                with open(ref_path, 'r', encoding='utf-8') as f:
                    references.append(f.read())
            continue

        if not os.path.isfile(ref_path):
            print(f"⚠️  {fname}: No reference summary, skipping")
            continue

        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_text = f.read()
        with open(ref_path, 'r', encoding='utf-8') as f:
            ref = f.read()

        doc_tokens = tokenizer.encode(doc_text)
        print(f"\n📄 Processing {fname} ({len(doc_tokens)} tokens)")

        # Run enhanced Map-Reduce with integrated critique
        summary = await summarize_document_mapreduce_with_critique(
            doc_text, mapreduce_app, critique_app, text_splitter,
            max_critique_iterations=2
        )

        # Clean and save
        summary = clean_thinking_tokens(summary)
        with open(gen_path, 'w', encoding='utf-8') as f:
            f.write(summary)

        print(f"✅ Saved to {gen_path}")

        generated_summaries.append(summary)
        references.append(ref)

    # Evaluation
    if generated_summaries and references:
        print(f"\n📊 EVALUATION RESULTS")
        print("=" * 30)

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

        P, R, F1 = bert_score(references, generated_summaries,
                              lang="vi", rescale_with_baseline=True)
        print("BERTScore P: {:.4f}, R: {:.4f}, F1: {:.4f}".format(
            P.mean().item(), R.mean().item(), F1.mean().item()
        ))


if __name__ == "__main__":
    asyncio.run(main())
