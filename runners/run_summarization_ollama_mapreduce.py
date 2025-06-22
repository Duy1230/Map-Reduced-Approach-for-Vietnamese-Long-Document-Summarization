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
from langsmith import traceable


class OllamaLLM(LLM):
    """Custom LLM wrapper for Ollama API"""

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
        return resp.json()["response"]

    async def _acall(self, prompt: str, stop=None, run_manager=None, **kwargs):
        return self._call(prompt, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def get_num_tokens(self, text: str) -> int:
        # Simple approximation - could be improved with proper tokenizer
        return len(text.split())


# Map-Reduce State Definitions
class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


class SummaryState(TypedDict):
    content: str


def create_map_reduce_graph(llm, token_max: int = 1000):
    """Create the Map-Reduce graph for summarization"""

    # Vietnamese prompts similar to the original script
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một chuyên gia tóm tắt nội dung.
Vui lòng viết một bản tóm tắt chi tiết cho đoạn văn bản sau bằng **tiếng Việt**.

{content}

Lưu ý: Không sử dụng dấu đầu dòng, hãy viết bằng câu đầy đủ và theo đoạn văn.""")
    ])

    reduce_template = """
Sau đây là một tập hợp các bản tóm tắt:
{docs}

Hãy tổng hợp và chắt lọc chúng thành một bản tóm tắt cuối cùng, toàn diện về các chủ đề chính bằng tiếng Việt.
Không sử dụng dấu đầu dòng, hãy viết bằng câu đầy đủ và theo đoạn văn.
"""

    reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

    def length_function(documents: List[Document]) -> int:
        """Get number of tokens for input contents."""
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

    async def _reduce(docs_input) -> str:
        if isinstance(docs_input, list):
            docs_text = "\n\n".join([doc.page_content for doc in docs_input])
        else:
            docs_text = str(docs_input)

        prompt = reduce_prompt.invoke({"docs": docs_text})
        response = await llm.ainvoke(prompt.messages[0].content)
        return response

    # Collapse summaries if they're too long
    async def collapse_summaries(state: OverallState):
        doc_lists = split_list_of_docs(
            state["collapsed_summaries"], length_function, token_max
        )
        results = []
        for doc_list in doc_lists:
            result = await acollapse_docs(doc_list, _reduce)
            # Ensure result is a string, not a Document
            if isinstance(result, Document):
                content = result.page_content
            else:
                content = str(result)
            results.append(Document(page_content=content))

        return {"collapsed_summaries": results}

    # Determine if we should collapse summaries or generate final summary
    def should_collapse(
        state: OverallState,
    ) -> Literal["collapse_summaries", "generate_final_summary"]:
        num_tokens = length_function(state["collapsed_summaries"])
        if num_tokens > token_max:
            return "collapse_summaries"
        else:
            return "generate_final_summary"

    # Generate the final summary
    async def generate_final_summary(state: OverallState):
        response = await _reduce(state["collapsed_summaries"])
        # Ensure response is a string, not a Document
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


async def summarize_document_mapreduce(doc_text: str, app, text_splitter, chunk_size: int = 1000):
    """Summarize a document using Map-Reduce approach"""

    # Create document and split it
    doc = Document(page_content=doc_text)
    split_docs = text_splitter.split_documents([doc])

    print(f"Split document into {len(split_docs)} chunks")

    # Run the Map-Reduce graph
    result = None
    async for step in app.astream(
        {"contents": [doc.page_content for doc in split_docs]},
        {"recursion_limit": 10},
    ):
        if "generate_final_summary" in step:
            result = step["generate_final_summary"]["final_summary"]

    return result


async def main():
    OLLAMA_URL = "http://localhost:11434"
    MODEL_NAME = "llama3.2:3b"  # You can change this to other models

    # Initialize LLM
    llm = OllamaLLM(ollama_url=OLLAMA_URL,
                    model_name=MODEL_NAME, max_new_tokens=2048)

    # Initialize text splitter
    text_splitter = CharacterTextSplitter(
        chunk_size=12000,
        chunk_overlap=200,
        separator="\n\n"
    )

    # Create Map-Reduce graph
    app = create_map_reduce_graph(llm, token_max=10000)

    # Tokenizer for evaluation (keeping the same as original script)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b")

    docs_dir = "data/doc"
    summary_dir = "data/summary"
    generated_summaries_dir = "data/generated_summaries_mapreduce"

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

        # Generate summary using Map-Reduce
        summary = await summarize_document_mapreduce(doc_text, app, text_splitter)

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
