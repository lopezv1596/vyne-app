import subprocess
import json
import requests
from bs4 import BeautifulSoup
import psutil
import os
import random
import datetime
import sqlite3  # ✅ Use SQLite instead of JSON for fast memory handling
import threading
from llama_cpp import Llama
import sys
from flask import Flask, request, jsonify
import gradio as gr

sys.path.append('/Volumes/Vyne_SSD/vyne_brain/')
from add_commands import search_wikipedia

# ✅ Define model paths for LLaMA and Mistral
MODEL_PATH_LLAMA = "/Users/vincentscomputer/.cache/huggingface/download/vicuna-7b-v1.5-16k.Q6_K.gguf"
MODEL_PATH_MISTRAL = "/Users/vincentscomputer/Desktop/vyne-backend/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# ✅ Load both models with optimized settings
llm_llama = Llama(model_path=MODEL_PATH_LLAMA, n_gpu_layers=2, n_ctx=2048)
llm_mistral = Llama(model_path=MODEL_PATH_MISTRAL, n_gpu_layers=2, n_ctx=2048)

app = Flask(__name__)

def chat_with_vyne(user_input):
    try:
        result = subprocess.run(
            ["python3", "/Users/vincentscomputer/Desktop/vyne-backend/vyne_chat.py"],
            input=user_input,  # ✅ No need to encode
            capture_output=True,
            text=True
        )
        return result.stdout.strip() if result.stdout else "⚠️ Vyne did not return a response."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


@app.route("/api/vyne", methods=["POST"])
def api_vyne():
    data = request.json
    user_input = data.get("input", "").strip()
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    response = chat_with_vyne(user_input)
    return jsonify({"response": response})

def launch_gradio():
    interface = gr.Interface(
        fn=chat_with_vyne,
        inputs="text",
        outputs="text",
        title="Vyne AI Web Interface",
        description="Type your request below, and Vyne will generate a response."
    )
    interface.launch(share=True)

# Run Flask and Gradio together
if __name__ == "__main__":
    threading.Thread(target=launch_gradio).start()
    app.run(host="0.0.0.0", port=5000, debug=True)
