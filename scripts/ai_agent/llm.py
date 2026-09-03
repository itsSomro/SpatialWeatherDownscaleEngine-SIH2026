import os
import sys
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Warning: HF_TOKEN is not set. AI responses may not work.", file=sys.stderr)

endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.5-9B",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.3,
)

llm = ChatHuggingFace(llm=endpoint)
