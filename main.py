from fastapi import FastAPI
import requests
import os
from pydantic import BaseModel

app = FastAPI()

# ==============================
# 🔑 環境変数（Renderで設定）
# ==============================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_AI_NAME = os.getenv("AI_NAME", "TSURUKAME CORE")

# ==============================
# 📦 データモデル
# ==============================

class AIRequest(BaseModel):
    text: str
    user_id: str | None = None
    ai_name: str | None = None   # ← ユーザー別に上書き可能

# ==============================
# 🌌 Gemini 呼び出し関数
# ==============================

def call_gemini(prompt: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    r = requests.post(url, json=data)
    return r.json()

# ==============================
# 🌐 API
# ==============================

@app.get("/")
def root():
    return {"status": f"{DEFAULT_AI_NAME} online"}

# 🧠 思考AI
@app.post("/ai")
def ai(req: AIRequest):

    ai_name = req.ai_name if req.ai_name else DEFAULT_AI_NAME

    system_prompt = f"""
あなたは「{ai_name}」。
ユーザーの思考を拡張し、未来を示すAI。

ユーザーID: {req.user_id}

ユーザー入力:
{req.text}
"""

    return call_gemini(system_prompt)

# 👤 プロファイル推定AI
@app.post("/profile")
def profile(req: AIRequest):

    prompt = f"""
次の文章から年齢層と性別を推定し、JSONだけで返せ。
例:
{{"age":"20s","gender":"male"}}
