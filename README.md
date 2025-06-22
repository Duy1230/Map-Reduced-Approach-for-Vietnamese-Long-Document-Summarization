# Vietnamese Document Summarization

Comparing 5 different ways to summarize long Vietnamese documents using AI.

## What's This About?

We built 5 different approaches to see which works best:

1. **Truncated** - Just cut the document to fit
2. **Map-Reduce** - Split into chunks, summarize each, then combine
3. **Map-Reduce + Critique** - Same as above but the AI reviews its own work
4. **Iterative** - Keep improving the summary step by step
5. **Hierarchical** - Use the document's structure (headers, sections) to summarize smartly

## Getting Started

### 1. Get the code and data
```bash
git clone <repository-url>
cd GD1-project
```

Download our Vietnamese documents from [Google Drive](https://drive.google.com/your-drive-link-here) and put them in a `data_1/` folder.

### 2. Install Ollama (the AI runner)
Get it from [ollama.ai](https://ollama.ai/), then:
```bash
ollama serve
ollama pull llama3.2:3b
```

### 3. Install Python stuff
```bash
pip install -r requirements.txt
```

### 4. Optional: Add your OpenAI key
Create a `.env` file if you want better evaluation metrics:
```
OPENAI_API_KEY=your_key_here
```

## How to Use

### Try the demo (easiest way)
```bash
streamlit run streamlit_demo.py
```
This opens a web interface where you can upload documents and see all 5 methods compared side-by-side.

### Run full evaluation
Compare all methods on your entire dataset:
```bash
# Test one approach
python run_full_evaluation_pipeline.py --approach mapreduce

# Test hierarchical with custom depth
python run_full_evaluation_pipeline.py --approach mapreduce_hierarchical --max-depth 2

# Just test on 5 documents first
python run_full_evaluation_pipeline.py --approach iterative --max-samples 5
```

## How We Measure Quality

- **ROUGE** - Checks how many words match between our summary and the "gold standard"
- **BERTScore** - Uses AI to check if the meaning is similar
- **Semantic Similarity** - Measures how close the summaries are conceptually

