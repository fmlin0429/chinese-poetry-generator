"""
Chinese Poetry Chatbot - Gradio Web App
=======================================

A web interface for the fine-tuned TinyLlama model that generates 
classical Chinese five-character poetry (五言诗).

Deployment: Hugging Face Spaces
Model: TinyLlama-1.1B-Chat-v1.0 + LoRA fine-tuning
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model components
tokenizer = None
model = None
chat_pipeline = None

# Model configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./tinyllama-poetry"  # Path to your LoRA adapter
SYSTEM_PROMPT = "You are a wise ancient Chinese poet. Respond only in five-character classical lines (五言诗). Each line should have exactly 5 Chinese characters."

def load_model_and_tokenizer():
    """
    Load the fine-tuned model with LoRA adapter.
    This function handles the initial loading with proper error handling.
    """
    global tokenizer, model, chat_pipeline
    
    try:
        logger.info("Starting model loading...")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapter if available
        if os.path.exists(ADAPTER_PATH):
            logger.info("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            logger.info("LoRA adapter loaded successfully!")
        else:
            logger.warning(f"LoRA adapter not found at {ADAPTER_PATH}. Using base model.")
            model = base_model
        
        # Create pipeline
        chat_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def generate_poem(user_input, history):
    """
    Generate a Chinese poem based on user input.
    
    Args:
        user_input (str): User's topic or prompt
        history (list): Chat history (for Gradio ChatInterface)
    
    Returns:
        str: Generated poem or error message
    """
    global chat_pipeline, tokenizer
    
    if chat_pipeline is None:
        return "⚠️ 模型正在加载中，请稍等... (Model is loading, please wait...)"
    
    try:
        # Prepare the conversation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Please write a Chinese five-character poem about: {user_input}"}
        ]
        
        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        logger.info(f"Generating poem for: {user_input}")
        
        # Generate response
        with torch.no_grad():
            outputs = chat_pipeline(
                formatted_prompt,
                max_length=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False,
                num_return_sequences=1
            )
        
        # Extract the generated text
        generated_text = outputs[0]['generated_text'].strip()
        
        # Clean up the response
        if generated_text:
            # Remove any remaining special tokens or unwanted text
            generated_text = generated_text.replace("<|assistant|>", "").strip()
            return generated_text
        else:
            return "抱歉，无法生成诗歌。请尝试其他主题。(Sorry, unable to generate poem. Please try another topic.)"
            
    except Exception as e:
        logger.error(f"Error generating poem: {e}")
        return f"生成过程中出现错误: {str(e)} (Error during generation: {str(e)})"

def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .header {
        text-align: center;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 12px;
        color: #666;
    }
    """
    
    # Create the chat interface
    with gr.Blocks(css=custom_css, title="中国古诗生成器 | Chinese Poetry Generator") as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>🏮 中国古诗生成器 | Chinese Poetry Generator</h1>
            <p><strong>基于 TinyLlama-1.1B 微调的五言诗生成模型</strong></p>
            <p><em>Fine-tuned TinyLlama-1.1B for Classical Chinese Poetry Generation</em></p>
        </div>
        """)
        
        # Instructions
        with gr.Accordion("📖 使用说明 | Instructions", open=False):
            gr.Markdown("""
            ### 中文说明：
            - 输入任何主题（如：春天、月亮、山水），模型将生成相应的五言诗
            - 五言诗格式：每行5个汉字，通常4行组成一首诗
            - 示例主题：春天、夏日、秋风、冬雪、花朵、月亮、山川、友情
            
            ### English Instructions:
            - Enter any topic (e.g., spring, moon, mountains), and the model will generate corresponding classical Chinese poetry
            - Five-character format: 5 Chinese characters per line, typically 4 lines per poem  
            - Example topics: spring, summer, autumn, winter, flowers, moon, mountains, friendship
            """)
        
        # Chat interface
        chatbot = gr.ChatInterface(
            fn=generate_poem,
            title="",
            description="",
            examples=[
                "春天 (Spring)",
                "月亮 (Moon)", 
                "山水 (Mountains and Rivers)",
                "花朵 (Flowers)",
                "友情 (Friendship)",
                "夏日 (Summer)",
                "秋风 (Autumn Wind)",
                "雪花 (Snowflakes)"
            ],
            cache_examples=False,
            retry_btn="🔄 重试 | Retry",
            undo_btn="↩️ 撤销 | Undo",
            clear_btn="🗑️ 清空 | Clear",
            submit_btn="✨ 生成诗歌 | Generate Poem",
            textbox=gr.Textbox(
                placeholder="请输入诗歌主题... | Enter your topic...",
                container=False,
                scale=7
            )
        )
        
        # Footer with model info
        gr.HTML("""
        <div class="footer">
            <p>🤖 <strong>Powered by TinyLlama-1.1B</strong> | 
               🎨 <strong>Fine-tuned for Chinese Poetry</strong> | 
               🚀 <strong>Deployed on Hugging Face Spaces</strong></p>
            <p><em>Model fine-tuned using LoRA (Low-Rank Adaptation) for efficient training</em></p>
        </div>
        """)
        
        # Loading status
        with gr.Row():
            status = gr.HTML("""
            <div style="text-align: center; padding: 10px;">
                <p>🔄 <strong>模型加载中... | Loading model...</strong></p>
                <p><em>首次启动需要下载模型文件，请耐心等待 | First startup requires downloading model files, please wait patiently</em></p>
            </div>
            """)
    
    return demo, status

def update_status(status_component, message):
    """Update the status message"""
    return gr.HTML(f'<div style="text-align: center; padding: 10px;"><p>{message}</p></div>')

def main():
    """Main function to launch the app"""
    logger.info("Starting Chinese Poetry Chatbot...")
    
    # Create interface
    demo, status = create_interface()
    
    # Load model in the background
    def load_model_with_status():
        success = load_model_and_tokenizer()
        if success:
            return gr.HTML("""
            <div style="text-align: center; padding: 10px; color: green;">
                <p>✅ <strong>模型加载完成！| Model loaded successfully!</strong></p>
                <p><em>现在可以开始生成诗歌了 | Ready to generate poetry now</em></p>
            </div>
            """)
        else:
            return gr.HTML("""
            <div style="text-align: center; padding: 10px; color: red;">
                <p>❌ <strong>模型加载失败 | Model loading failed</strong></p>
                <p><em>请检查模型文件 | Please check model files</em></p>
            </div>
            """)
    
    # Load model when the app starts
    demo.load(load_model_with_status, outputs=status)
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )

if __name__ == "__main__":
    main()