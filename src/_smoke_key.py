from dotenv import load_dotenv, find_dotenv
import os
import google.generativeai as genai

# Carrega o .env da raiz do projeto, forçando override
load_dotenv(find_dotenv(filename=".env"), override=True)

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
print("Tem chave no env?", bool(api_key))
print("Prefixo/sufixo:", (api_key or "")[:6], (api_key or "")[-6:])

genai.configure(api_key=api_key)
res = genai.embed_content(model="text-embedding-004", content="ping")
print("Dimensão do embedding:", len(res["embedding"]))