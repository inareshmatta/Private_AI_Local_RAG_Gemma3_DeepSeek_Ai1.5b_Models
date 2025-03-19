# INM's Private AI - Document Analysis System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Model: Transformers](https://img.shields.io/badge/Model-Transformers-green.svg)](https://huggingface.co/transformers/)
[![GPU: CUDA](https://img.shields.io/badge/GPU-CUDA-red.svg)](https://developer.nvidia.com/cuda-zone)
[![Embeddings: Sentence-BERT](https://img.shields.io/badge/Embeddings-SBERT-orange.svg)](https://www.sbert.net/)
[![Database: FAISS](https://img.shields.io/badge/Database-FAISS-blueviolet.svg)](https://faiss.ai/)
[![NLP: Hugging Face](https://img.shields.io/badge/NLP-HuggingFace-yellow.svg)](https://huggingface.co/)
[![License: Open Source](https://img.shields.io/badge/License-OpenSource-brightgreen.svg)](https://opensource.org/)
[![Model: DeepSeek](https://img.shields.io/badge/Model-DeepSeek_R1_Qwen_1.5B-blue.svg)](https://huggingface.co/deepseek-ai)
[![Model: Gemma](https://img.shields.io/badge/Model-Gemma_3_1B_IT-red.svg)](https://ai.google.dev/gemma)


A private document analysis system using Retrieval-Augmented Generation (RAG) architecture. Supports PDF, DOCX, and PPTX files with state-of-the-art language models.

![System Architecture](https://via.placeholder.com/800x400.png?text=System+Architecture+Diagram)

## Features

- **Multi-Format Support**: Process PDF, DOCX, and PPTX documents
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Advanced NLP**:
  - Two model options (DeepSeek 1.5B or Gemma 3B)
  - Semantic search with FAISS vector indexing
  - Context-aware generation
- **Enterprise Security**: Local processing only, no data leaves your environment

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/private-ai.git
cd private-ai

# Install dependencies
pip install -r requirements.txt

# Download models (example path)
[deepseekr1](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
[gemma3](https://huggingface.co/google/gemma-3-1b-it)
```

## Usage

```bash
# Start the application
streamlit run app.py
```

### Basic Workflow

1. Upload documents through the sidebar.
2. Wait for processing completion (see progress indicators).
3. Ask questions in natural language.
4. View responses with source context.

## Configuration

### Model Selection
Edit `config.yaml`:

```yaml
models:
  default: "DeepSeek-R1-Distill-Qwen-1.5B"
  paths:
    DeepSeek: "./models/DeepSeek-R1-Distill-Qwen-1.5B"
    Gemma: "./models/gemma-3-1b-it"
```

### Advanced Settings

Modify `app.py`:

```python
class DocumentRAG:
    def __init__(self):
        self.chunk_size = 1500  # Context window size
        self.overlap = 200      # Chunk overlap
        self.top_k = 5          # Retrieved context chunks
```

## Technical Specifications

### Document Processing Pipeline

| Stage             | Technology            | Description                     |
|------------------|----------------------|---------------------------------|
| Text Extraction  | PyMuPDF               | Fast PDF text extraction       |
| Chunking        | Sentence Transformers | Semantic text segmentation     |
| Embedding       | MiniLM-L6             | 384-dimension embeddings       |
| Indexing        | FAISS-IVF             | Efficient similarity search    |
| Generation      | Causal LMs            | Context-aware responses        |

### Performance Metrics

| Operation        | CPU (i9)  | GPU (V100)  |
|-----------------|----------|------------|
| PDF Processing  | 2.1s/page | 1.8s/page  |
| Embedding       | 45ms/chunk | 12ms/chunk |
| Query Response  | 3-5s      | 0.8-1.2s   |

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
- Reduce `chunk_size` in config.
- Use smaller batch sizes.

#### 2. Missing Dependencies

```bash
pip install --force-reinstall -r requirements.txt
```

#### 3. Model Loading Errors
- Verify model file integrity.
- Check Hugging Face token permissions.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Pull requests welcome! Please follow our [contribution guidelines](CONTRIBUTING.md).

---
**Note**: Requires a minimum of 8GB VRAM for GPU operation with default models.
