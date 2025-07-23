
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import faiss
import requests

API_KEY = "sk-zk2aa3ba45720912b2db6e9aac5104d931a687f5fcc20ef7"  # 智增增 key
BASE_URL = "https://api.zhizengzeng.com/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# 加载向量库
texts = [line.strip() for line in open("ziwei_rag_text_blocks.txt", "r", encoding="utf-8")]
embeddings = np.load("ziwei_embeddings.npy").astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# FastAPI app
app = FastAPI()

class Query(BaseModel):
    question: str

def get_embedding(text):
    resp = requests.post(f"{BASE_URL}/embeddings", headers=HEADERS, json={
        "input": text,
        "model": "text-embedding-ada-002"
    })
    return resp.json()["data"][0]["embedding"]

def get_answer(prompt):
    resp = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json={
        "model": "gpt-4-1106-preview",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    })
    return resp.json()["choices"][0]["message"]["content"]

@app.post("/ziwei/ask")
def ask(query: Query):
    q_emb = np.array(get_embedding(query.question)).astype('float32')
    D, I = index.search(q_emb.reshape(1, -1), 3)
    blocks = [texts[i] for i in I[0]]
    context = "\n".join(blocks)
    prompt = f"{context}\n\n问题：{query.question}\n回答："
    answer = get_answer(prompt)
    return {"answer": answer}

