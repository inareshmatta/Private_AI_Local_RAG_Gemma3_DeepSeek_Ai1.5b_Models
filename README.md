# Private_AI_Local_RAG_Gemma3_DeepSeek_Ai1.5b_Models
Building a Private AI: A Comprehensive Look at a Retrieval-Augmented Generation (RAG) Streamlit Project

March 19, 2025
Project Link

Building a Private AI: A Comprehensive Look at a Retrieval-Augmented Generation (RAG) Streamlit Project


Our Retrieval-Augmented Generation (RAG) system is powered by a local language model that incorporates internal chain-of-thought reasoning. This means that, internally, the model works through intermediate steps and logical reasoning to generate a robust, context-aware response. However, to ensure clarity and simplicity for end users, our system is engineered to only display the final, concise answer.

Key Points:

Internal Reasoning: The model internally processes the context and question through multiple reasoning steps. This helps in formulating a detailed and accurate response.
Clean Output: Through our advanced prompt engineering and post-processing (using regex to remove any <think>...</think> markers), the system strips out the internal chain-of-thought details, showing only the final answer.
Benefits:

This approach allows our RAG system to function as a reasoning model—leveraging sophisticated internal processing while keeping the user experience straightforward and focused solely on the answer.



Table of Contents
Introduction
Understanding Retrieval-Augmented Generation (RAG)
Project Goals and High-Level Architecture
Key Components and Libraries
Document Ingestion and Text Extraction
Chunking and Embedding
FAISS for Similarity Search
Language Model and Post-Processing
Streamlit UI and Workflow
Detailed Walkthrough of the Code Project Structure Text Cleaning Helpers Post-Processing of Model Output The DocumentRAG Class Streamlit UI Logic
Deployment Considerations
Challenges and Future Enhancements
Conclusion
Sample LinkedIn Post

1. Introduction
In the rapidly evolving landscape of artificial intelligence (AI), Retrieval-Augmented Generation (RAG) has emerged as a powerful technique that combines document retrieval with language model generation to produce more accurate, context-aware answers. This article provides a deep dive into a Private AI RAG Streamlit project, explaining how each piece fits together and how you can build or extend such a system for your own needs.

We’ll explore document ingestion, embedding with SentenceTransformers, similarity search with FAISS, language model inference (on GPU if available), and the Streamlit user interface. Additionally, we’ll detail the post-processing steps that ensure the final AI response is free of chain-of-thought markers like <think>...</think>—making the system more user-friendly and production-ready.

This article is meant for engineers, data scientists, and AI enthusiasts looking to understand how to build a RAG system that respects user privacy and can run on local or on-premise hardware.




2. Understanding Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation is a technique that addresses one of the key limitations of large language models: context windows. Traditional LLMs rely on the text input you provide (the “prompt”) and their internal parameters, but they can’t directly “see” or “search” external documents. This often leads to hallucinations or incomplete answers, especially when the model’s training data is outdated or limited.

RAG mitigates this by introducing a retrieval step:

Index relevant documents (PDFs, DOCX, PPTX, or others) using embeddings.
Search for relevant chunks using a query.
Feed the retrieved chunks into the language model as context.
Generate a final answer that is informed by the specific content found in your local corpus.

By doing so, RAG ensures the AI system can provide up-to-date, context-specific responses without relying solely on the LLM’s parametric memory. This approach is especially beneficial for private or proprietary documents, as it keeps everything on your local machine or private server.








3. Project Goals and High-Level Architecture
3.1 Goals
Local AI: All processing remains on your own machine or server, protecting sensitive data.
Multiple Document Types: Ingest and index PDFs, DOCX, and PPTX files.
GPU Support: If available, leverage CUDA for faster inference.
Concise, Polished Answers: Post-process the model’s output to remove chain-of-thought and ensure a final, user-friendly response.
Streamlit UI: Provide an intuitive web-based chat interface.

3.2 High-Level Architecture
Document Ingestion:
Embedding and Indexing:
Query:
Generation:
Streamlit:



4. Key Components and Libraries
4.1 Streamlit
A Python library for creating interactive web applications.
In this project, Streamlit powers the chat interface, the sidebar for uploading documents, and the real-time display of model outputs.

4.2 pdfplumber, python-docx, python-pptx
pdfplumber: More advanced than PyPDF2, often better at extracting text from multi-column or complex PDFs.
python-docx: Allows reading .docx files to extract paragraph text.
python-pptx: Allows reading .pptx slides and shapes to extract textual content.

4.3 SentenceTransformers
Provides pre-trained embedding models (like all-mpnet-base-v2).
We encode each text chunk into a dense vector for similarity search.

4.4 FAISS
A library developed by Facebook AI Research for fast similarity search.
Stores embeddings in an index (here, IndexFlatL2), allowing quick retrieval of top-k similar chunks.

4.5 Hugging Face Transformers
We use AutoTokenizer, AutoModelForCausalLM, and AutoConfig to load and run a local language model.
If torch.cuda.is_available() is True, we run the model on GPU with half-precision (torch.float16) for speed and memory efficiency.

4.6 Python Standard Libraries
os: Handling file paths and system operations.
re: Regex for cleaning text and removing chain-of-thought tags.
tempfile: Creating temporary files for each upload.
numpy: Handling embeddings as arrays (FAISS uses NumPy arrays).



5. Document Ingestion and Text Extraction
Document ingestion is critical to any RAG system. The steps are:

Upload a file via Streamlit’s file_uploader.
Create a temporary file with the same extension to preserve the correct format.
Use specialized extraction libraries: pdfplumber for PDF python-docx for DOCX python-pptx for PPTX
Clean the text to remove page headers (e.g., “PAGE 1”), repeated characters, or other noise.

This ensures you get the highest quality text for embedding.



6. Chunking and Embedding
6.1 Why Chunk?
Large language models and retrieval systems often have a context window limit. If you feed them massive text, they may ignore or truncate it. Chunking ensures each chunk is small enough to handle but large enough to be meaningful.

6.2 Overlap
By having a slight overlap (e.g., chunk_size=1000 with overlap=100), you minimize the risk of cutting important information in half. Each chunk can still carry enough context.

6.3 Embedding
After chunking, we use SentenceTransformers (all-mpnet-base-v2 or another model) to generate embeddings:

Convert each chunk to a dense vector.
Store these vectors in FAISS, along with the raw chunk text in a Python list (doc_store).



7. FAISS for Similarity Search
FAISS (Facebook AI Similarity Search) is the library we use to index and search embeddings:

IndexFlatL2: The simplest FAISS index using L2 distance.
Index creation is done by specifying the dimensionality of embeddings (e.g., 768 for some BERT-based models).
Add: We insert embeddings of all text chunks.
Search: We query top-k nearest neighbors using index.search(query_vector, k).

This step is crucial because it retrieves the relevant chunks that the language model will use to answer questions. Without retrieval, the model might rely solely on its internal parameters and produce incomplete or incorrect answers.



8. Language Model and Post-Processing
8.1 Local Language Model
We rely on a local model (e.g., DeepSeek-R1 or Gemma-3). This approach ensures privacy—your data never leaves your environment. We use:

AutoConfig to load the model config.
AutoTokenizer for tokenization.
AutoModelForCausalLM for text generation.
If torch.cuda.is_available(), we place the model on the GPU with torch_dtype=torch.float16.

8.2 Prompt Construction
When a user asks a question:

We embed the question.
We retrieve top-k chunks from FAISS.
We build a prompt that includes: A short system instruction: “You are a helpful assistant…” The retrieved context. The user’s question. The phrase “Final Answer:” to indicate where generation should focus.

8.3 Removing lt;thinkgt;...lt;/thinkgt;
Some local models might output chain-of-thought text in <think> blocks. We do a regex removal step:

re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL) 
This ensures the final user sees only the direct answer, not the internal reasoning tokens.

8.4 Stripping “Final Answer:”
If the model echoes “Final Answer:”, we remove it so the user sees a clean answer.



9. Streamlit UI and Workflow
9.1 Sidebar
Model Selection: The user chooses between two models (DeepSeek or Gemma).
File Uploader: Accepts multiple PDF, DOCX, PPTX files.
Token Counters: Shows how many tokens have been used for input and output.

9.2 Main Chat Interface
Chat History: Each user message is displayed in one bubble, each AI response in another.
Chat Input: The user types a question.
Spinner: While the model processes, Streamlit shows a “Thinking…” message.
Response: The final answer is displayed as soon as it’s ready.

9.3 Under the Hood
Upload → Ingest → Embed → FAISS Index.
Query → Embed Query → Retrieve → Construct Prompt → Generate → Post-Process → Display.

10. Detailed Walkthrough of the Code
Below, we’ll break down the code from top to bottom, referencing the Python script (or Jupyter cells).

10.1 Project Structure
A typical layout might look like:

MyRAGProject/
├─ rag_app.py
├─ requirements.txt
├─ .gitignore
└─ README.md 
rag_app.py: The main Streamlit application code.
requirements.txt: Python dependencies (e.g., streamlit, pdfplumber, torch, sentence-transformers, etc.).
.gitignore: Ignores venvs, caches, etc.
README.md: Explains usage.

10.2 Text Cleaning Helpers
def collapse_repeated_chars(text: str, threshold=3) -> str:
    # ...
def clean_text(text: str) -> str:
    # ... 
collapse_repeated_chars uses a regex pattern (.)\1{2,} to find runs of 3+ identical characters and reduce them to 2.
clean_text filters lines containing “PAGE” or “©”, then calls collapse_repeated_chars.

10.3 Post-Processing Output
def post_process_response(raw_answer: str) -> str:
    # Remove <think> blocks
    processed = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
    # Remove "Final Answer:"
    processed = re.sub(r"(?i)^final answer:\s*", "", processed)
    return processed 
We specifically remove any chain-of-thought text and the literal “Final Answer:” prefix.

10.4 The DocumentRAG Class
Initialization:

Loads a Hugging Face model config from <MODEL_PATH>.
Optionally disables sliding window attention if disable_swa is True.
Loads the model with half-precision (torch.float16) if on GPU.
Creates a FAISS index with IndexFlatL2.

Document Extraction:

_extract_text_pdf, _extract_text_docx, _extract_text_pptx handle each file type.
_extract_text dispatches to the correct method based on file extension, then calls clean_text.

Chunking:

_chunk_text slices the text into segments of chunk_size=1000 characters with overlap=100.

Ingestion:

def ingest_document(self, file_path: str, original_filename: str):
    # 1. Extract text
    # 2. Chunk text
    # 3. Embed each chunk
    # 4. Add embeddings to FAISS
    # 5. Extend doc_store 
This method updates the FAISS index and local doc store so that future queries can retrieve these chunks.

Query:

def query(self, question: str, top_k: int = 3, max_length: int = 512) -> str:
    # 1. If no docs, return a message
    # 2. Embed the question
    # 3. FAISS search for top_k chunks
    # 4. Build prompt with context + question
    # 5. Generate answer with the LLM
    # 6. Post-process answer
    # 7. Return final cleaned answer 
Ensures each user query is answered with the best possible context from the user’s own documents.

10.5 Streamlit UI Logic
Initialization:

if "rag" not in st.session_state:
    st.session_state.rag = DocumentRAG() 
We store the DocumentRAG instance in session state so it persists across interactions.

File Uploader:

uploaded_files = st.file_uploader(..., accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        # Save temp file with correct extension
        # rag.ingest_document(...) 
We preserve the extension so _extract_text can detect file type.

Chat Input:

prompt = st.chat_input("Ask a question...")
if prompt:
    response = st.session_state.rag.query(prompt)
    # Display the answer 
A user can type any question, and the system calls rag.query(...) to generate a final response.

11. Deployment Considerations
11.1 Python Environment
Ensure you have the correct versions of:

PyTorch (GPU-enabled if you want CUDA)
transformers
sentence-transformers
faiss-cpu (or faiss-gpu if you have a suitable environment, though CPU is common)

11.2 Virtual Environments
Use conda or venv to isolate dependencies. This helps avoid conflicts, especially with GPU builds of PyTorch.

11.3 Docker
If you want a portable solution, containerize your app. A Dockerfile might install conda, faiss-cpu, torch, and run streamlit.

11.4 GPU Memory
Large models can require significant VRAM (8GB+). If you have a smaller GPU, consider using a smaller model or quantized approach (like 8-bit or 4-bit quantization).



12. Challenges and Future Enhancements
Chain-of-Thought: Even with regex removal, some models might produce extra text. Fine-tuning or more advanced prompting could help.
Larger Documents: For extremely large PDFs, you may need more sophisticated chunking or summarization.
Authentication and Privacy: If you deploy this app, ensure it’s accessible only to authorized users if your documents are sensitive.
Scaling: If you need to handle thousands of documents, consider a more advanced FAISS index or a GPU-based indexing approach.
Better PDF Handling: Some PDFs have complicated layouts or require OCR. Tools like Tesseract or pdfplumber with advanced settings can help.





13. Conclusion
Building a Private AI system that retrieves from local documents and generates answers with a local model is entirely feasible with open-source libraries:

Streamlit for a friendly interface
SentenceTransformers for embeddings
FAISS for vector search
Hugging Face Transformers for language generation

By combining these components, you have a Retrieval-Augmented Generation pipeline that ensures data privacy (since everything runs locally) and contextual answers (since the system references your specific documents).

We walked through:

Document ingestion and cleaning
Chunking and embedding
FAISS-based retrieval
Local language model inference (with GPU support)
Post-processing to remove chain-of-thought markers

Armed with this knowledge, you can adapt the project to various domains—legal documents, research papers, internal wikis, or any private text corpora. The code base is modular, letting you swap out embedding models or language models to suit your hardware and data constraints.








14.1 Additional Prompting Strategies
Prompt engineering is an ongoing field of experimentation. Some strategies to refine your model’s behavior include:

Explicit Instructions: Telling the model “Answer concisely” or “Use bullet points” can shape the output.
Context Summaries: If your document corpus is huge, you might summarize chunks before passing them to the LLM.
System vs. User vs. Assistant Roles: Some frameworks differentiate these roles in the prompt. For example, “system” messages set the AI’s personality or instructions, while “user” messages contain questions.

14.2 Advanced PDF Layout Handling
Some PDFs contain multi-column text, tables, or embedded images. pdfplumber does a good job in many cases, but you can:

Use Tesseract OCR for scanned documents.
Adjust pdfplumber parameters for columns.
Manually parse table structures if needed.

14.3 Indexing Large Corpora
When you have thousands or millions of chunks, the IndexFlatL2 approach might become slow or memory-intensive. Consider:

FAISS IVF (Inverted File Index) or HNSW for approximate search.
Splitting your corpus across multiple indexes if you want domain-based retrieval.
Using a GPU-based FAISS index if you have the hardware and need maximum speed.

14.4 Monitoring and Logging
To keep track of usage:

Log how many tokens are used per query.
Store user queries and final answers in a database (if privacy policies allow) to refine or debug the system.
Integrate with an analytics tool to measure average response times, user satisfaction, or query success rates.

14.5 Security and Access Control
When deploying a private AI, consider:

Authentication: Restrict the Streamlit app behind a login.
HTTPS: If hosting externally, secure it with TLS/SSL.
Containerization: Docker or Kubernetes for consistent, reproducible deployments.
Data Governance: Ensure you comply with any data privacy regulations in your region.






14.6 Potential Future Upgrades
Better Summarization: If chunks are too large, incorporate a summarization step using another model.
Cross-Encoder Reranking: After retrieving top-k chunks, use a cross-encoder to re-rank them for even more relevant context.
4-bit or 8-bit Quantization: If VRAM is limited, consider quantizing your local model to fit in smaller GPUs.
Multi-lingual Support: If you have documents in multiple languages, pick an embedding model that supports them, or use separate indexes.
UI Enhancements: Add advanced features like a “Document Explorer,” “Chunk Explorer,” or a “Confidence Score” display.

Downloading Local Models from Hugging Face (Online & Offline Methods)

To kickstart your private AI solution, the first step is to download pre-trained models from the Hugging Face Model Hub and store them locally. This ensures that all model data remains on-premise—essential for maintaining data privacy and optimizing inference speed. Online, you can leverage the Transformers library by using the from_pretrained method with a specified cache directory. For example, calling from_pretrained("model-identifier", cache_dir="<LOCAL_MODEL_DIRECTORY>") (with "model-identifier" replaced by the desired model such as DeepSeek-R1-Distill-Qwen-1.5B or Gemma-3-1b-it, and <LOCAL_MODEL_DIRECTORY> replaced by your chosen folder) will download the model automatically into that folder. Alternatively, you can use the Hugging Face CLI to download the model by running commands like huggingface-cli login followed by huggingface-cli download model-identifier, which gives you additional control over the download process.

For situations where you need to work completely offline or prefer to manually manage the files, you can also download the model directly from the Hugging Face Model Hub website. Simply navigate to your desired model’s page, and use the “Download” option to get the model files as a ZIP archive. Once downloaded, unzip the contents into a local directory of your choice. You can then load the model in your code using from_pretrained("<LOCAL_DIRECTORY>"), where <LOCAL_DIRECTORY> points to the folder containing the unzipped model files. This offline approach not only enables you to work in environments without internet access but also gives you complete control over the model version and storage, ensuring that all components of your AI system remain secure and on-premise.

Final Thoughts
By following this comprehensive guide, you can stand up a private AI system that harnesses the power of large language models while keeping data firmly under your control. The combination of FAISS for retrieval, SentenceTransformers for embeddings, Hugging Face for generation, and Streamlit for UI is flexible, powerful, and relatively straightforward to customize.

Whether you’re building an internal knowledge base, assisting with legal document Q&A, or simply exploring the latest in AI—this RAG architecture is a solid foundation for advanced use cases. As the AI ecosystem continues to evolve, you can swap out or update each layer (embedding model, LLM, or UI) to keep your system at the cutting edge.

