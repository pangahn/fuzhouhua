# Fuzhou Dialect Speech Recognition

An open-source project dedicated to fine-tuning a Whisper-based speech recognition model for the [Fuzhou dialect](https://en.wikipedia.org/wiki/Fuzhou_dialect) (福州话).

## 🚀 Overview

This project builds a robust ASR (Automatic Speech Recognition) system for the Fuzhou dialect—a regional Chinese language with limited existing NLP support. We leverage OpenAI's Whisper architecture to create an end-to-end pipeline from data collection to model deployment.

## 🎯 Key Features

- **Comprehensive Dataset** - Professionally curated audio samples with accurate transcriptions
- **State-of-the-Art Models** - Fine-tuned Whisper variants optimized for the Fuzhou dialect
- **User-Friendly Interface** - Interactive web application for immediate speech recognition

## 📋 Project Milestones

### Dataset Construction
- [x] Collect diverse Fuzhou dialect audio/video content
- [x] Implement OCR workflow for subtitle extraction and alignment
- [x] Release curated training dataset: [i18nJack/fuzhouhua](https://huggingface.co/datasets/i18nJack/fuzhouhua)

### Model Development
- [x] Fine-tune Whisper architecture for Fuzhou dialect recognition
- [ ] Release models in various sizes for different application requirements
- [ ] Optimize for latency and resource-constrained environments

### Deployment
- [ ] Develop web-based interface for real-time speech recognition
- [ ] Create API endpoints for third-party integration
- [ ] Publish comprehensive documentation and usage examples

## 🛠️ Quick Start

### Installation

```bash
# Clone the repository
git clone git@github.com:pangahn/fuzhouhua.git
cd fuzhouhua

# Install dependencies
uv sync
```

### Configuration

Create a `.env` file in the project root:

```dotenv
# OCR configuration
OCR_OPENAI_BASE_URL=https://openrouter.ai/api/v1
OCR_OPENAI_API_KEY=your_openai_key

# OpenAI API configuration
OPENAI_BASE_URL=https://api.moonshot.cn/v1
MODEL_NAME=moonshot-v1-8k
OPENAI_API_KEY=your_moonshot_key

# Hugging Face access
HF_TOKEN=hf_your_token
```

## 💻 Development Environment

### Prerequisites
- macOS or Linux (Windows users can use WSL)
- Python 3.10+
- [Homebrew](https://brew.sh) (for macOS users)

### Setup Instructions

```bash
# Install uv package manager
brew install uv

# Initialize project environment
uv init fuzhouhua --python=3.10

# Add required dependencies
uv add paddle2onnx==1.3.1
```

## 🔤 PaddleOCR Integration

Convert PaddleOCR models to ONNX format for wider compatibility:

```bash
# Detection Model
paddle2onnx \
  --model_dir ./models/ocr/pdmodel/ch_PP-OCRv4_det_server_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./models/ocr/onnxmodel/ch_PP-OCRv4_det_server_infer.onnx \
  --opset_version 11 \
  --enable_onnx_checker True

# Recognition Model
paddle2onnx \
  --model_dir ./models/ocr/pdmodel/ch_PP-OCRv4_rec_server_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./models/ocr/onnxmodel/ch_PP-OCRv4_rec_server_infer.onnx \
  --opset_version 11 \
  --enable_onnx_checker True

# Classification Model
paddle2onnx \
  --model_dir ./models/ocr/pdmodel/ch_ppocr_mobile_v2.0_cls_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./models/ocr/onnxmodel/ch_ppocr_mobile_v2_cls.onnx \
  --opset_version 11 \
  --enable_onnx_checker True
```

## 🤝 Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md) before getting started.

## 📝 Citation

If you use this project in your research or applications, please cite:

```bibtex
@software{fuzhouhua_asr,
  author = {Pan, Gahn},
  title = {Fuzhou Dialect Speech Recognition},
  year = {2025},
  url = {https://github.com/pangahn/fuzhouhua}
}
```

## 📄 License

This project is licensed under the [GNU General Public License v3.0](LICENSE) (GPL-3.0).