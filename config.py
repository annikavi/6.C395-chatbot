import os
from dotenv import load_dotenv

load_dotenv()

# Default inference model — swap for any HuggingFace model ID
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Set to your fine-tuned model ID if you upload one to HuggingFace Hub
MY_MODEL = None

HF_TOKEN = os.getenv("HF_TOKEN")
