# 🏮 Chinese Poetry Generator | 中国古诗生成器

A web application that generates classical Chinese five-character poetry (五言诗) using a fine-tuned TinyLlama-1.1B model with LoRA adaptation.

## 🌟 Features

- **Classical Chinese Poetry Generation**: Creates authentic-style 五言诗 (5-character per line poems)
- **Interactive Web Interface**: User-friendly chat interface built with Gradio
- **Bilingual Support**: Interface supports both Chinese and English
- **Fine-tuned Model**: TinyLlama-1.1B enhanced with LoRA for poetry generation
- **Easy Deployment**: Ready for Hugging Face Spaces deployment

## 🚀 Quick Start on Hugging Face Spaces

### Method 1: Direct Upload (Recommended)

1. **Create a new Space on Hugging Face**:
   - Go to https://huggingface.co/new-space
   - Choose a name like `chinese-poetry-generator`
   - Select **Gradio** as the SDK
   - Set visibility as desired (Public/Private)

2. **Upload your files**:
   ```
   your-space/
   ├── app.py                 # Main application file
   ├── requirements.txt       # Dependencies
   ├── README.md             # This file
   └── tinyllama-poetry/     # Your LoRA adapter folder
       ├── adapter_config.json
       ├── adapter_model.safetensors
       └── ...
   ```

3. **Your Space will automatically build and deploy!**

### Method 2: Git Clone and Push

1. **Clone your new Space repository**:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   cd YOUR_SPACE_NAME
   ```

2. **Copy files to the repository**:
   ```bash
   # Copy the application files
   cp /path/to/your/app.py .
   cp /path/to/your/requirements.txt .
   cp /path/to/your/README.md .
   
   # Copy your fine-tuned model adapter
   cp -r /path/to/your/tinyllama-poetry ./
   ```

3. **Push to Hugging Face**:
   ```bash
   git add .
   git commit -m "Add Chinese poetry generator app"
   git push
   ```

## 📁 Required Files

### 1. `app.py`
The main Gradio application file (already created).

### 2. `requirements.txt`
Dependencies for the application:
```
gradio>=4.0.0
transformers>=4.35.0
torch>=2.0.0
peft>=0.6.0
accelerate>=0.24.0
```

### 3. `tinyllama-poetry/` folder
Your fine-tuned LoRA adapter files:
- `adapter_config.json`
- `adapter_model.safetensors`
- `README.md` (optional)

## 🛠️ Local Development

### Prerequisites
- Python 3.8+
- Your fine-tuned LoRA adapter in `./tinyllama-poetry/`

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The app will be available at `http://localhost:7860`

## 🎯 Usage

1. **Enter a topic** in Chinese or English (e.g., "春天", "月亮", "mountains")
2. **Click "Generate Poem"** to create a classical Chinese poem
3. **View the result** in traditional 五言诗 format

### Example Topics:
- 春天 (Spring)
- 月亮 (Moon)  
- 山水 (Mountains and Rivers)
- 花朵 (Flowers)
- 友情 (Friendship)
- 夏日 (Summer)
- 秋风 (Autumn Wind)
- 雪花 (Snowflakes)

## 📊 Model Information

- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: Classical Chinese poetry examples
- **Task**: Chinese five-character poetry generation
- **Parameters**: ~1.1B base + ~12M trainable LoRA parameters

## 🚨 Troubleshooting

### Model Loading Issues
- Ensure `tinyllama-poetry/` folder contains all adapter files
- Check that the adapter was trained with compatible PEFT version
- Verify sufficient system memory (4GB+ recommended)

### Generation Problems
- Try different topics if output is poor
- Adjust temperature parameter (0.5-1.0 range)
- Check that system prompt is appropriate for your use case

## 📄 License

This project is open-source. The TinyLlama base model follows its respective license terms.

---

**Powered by TinyLlama-1.1B** | **Fine-tuned for Chinese Poetry** | **Deployed on Hugging Face Spaces**