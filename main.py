from fastapi import FastAPI
import requests
import os
from pydantic import BaseModel

app = FastAPI()

# Gemini API Key（Renderで設定）
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def call_gemini(prompt: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    r = requests.post(url, json=data)
    return r.json()

# ========= データモデル =========
class Question(BaseModel):
    text: str

# ========= API =========

@app.get("/")
def root():
    return {"status": "tsuruta-ai online"}

# 🧠 思考AI
@app.post("/ai")
def ai(q: Question):
    system_prompt = f"""
あなたは「つるたAI」。
ユーザーの思考を拡張し、未来を示すAI。

ユーザー入力:
{q.text}
"""
    return call_gemini(system_prompt)

# 👤 プロファイル推定AI（年齢・性別）
@app.post("/profile")
def profile(q: Question):
    prompt = f"""
次の文章から年齢層と性別を推定し、JSONだけで返せ。
例:
{{"age":"20s","gender":"male"}}

文章:
{q.text}
"""
    return call_gemini(prompt)

# ⚖️ 判断AI（分身AI）
@app.post("/judge")
def judge(q: Question):
    prompt = f"""
あなたはユーザーの分身AI。
次の問いに対して最適な判断を短く出せ。

問い:
{q.text}
"""
    return call_gemini(prompt)