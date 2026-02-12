from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# ==============================
# 🔑 環境変数（Renderで設定）
# ==============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_AI_NAME = os.getenv("AI_NAME", "TSURUKAME CORE")

# ここを gemini-2.0-flash-lite にする（将来差し替えできるよう環境変数対応）
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")  # :contentReference[oaicite:1]{index=1}

# ==============================
# 🌐 CORS（つるかめポータル等の別ドメインから叩く用）
# 本番では allow_origins を自分のドメインに絞るの推奨
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 📦 リクエストモデル
# ==============================
class AIRequest(BaseModel):
    text: str
    user_id: str | None = None
    ai_name: str | None = None  # ユーザーごとに上書き可能（将来カスタム用）

# ==============================
# 🌌 Gemini 呼び出し
# ==============================
def call_gemini(prompt: str):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")

    # generateContent endpoint（APIキーはヘッダーで渡すのが安全）
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
    }

    r = requests.post(url, headers=headers, json=data, timeout=60)

    # Gemini側のエラーを見える化
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    return r.json()

# ==============================
# 🌐 API
# ==============================
@app.get("/")
def root():
    return {"status": f"{DEFAULT_AI_NAME} online", "model": GEMINI_MODEL}

# 🧠 思考AI
@app.post("/ai")
def ai(req: AIRequest):
    ai_name = req.ai_name or DEFAULT_AI_NAME
    system_prompt = f"""あなたは「{ai_name}」。
ユーザーの思考を拡張し、未来を示すAI。

ユーザーID: {req.user_id}

ユーザー入力:
{req.text}
"""
    return call_gemini(system_prompt)

# 👤 プロファイル推定AI（年齢・性別）
@app.post("/profile")
def profile(req: AIRequest):
    prompt = f"""次の文章から年齢層と性別を推定し、JSONだけで返せ。
例:
{{"age":"20s","gender":"male"}}

文章:
{req.text}
"""
    return call_gemini(prompt)

# ⚖️ 判断AI（分身AI）
@app.post("/judge")
def judge(req: AIRequest):
    ai_name = req.ai_name or DEFAULT_AI_NAME
    prompt = f"""あなたは「{ai_name}」の分身AI。
次の問いに対して、最適な判断を短く出せ。

ユーザーID: {req.user_id}

問い:
{req.text}
"""
    return call_gemini(prompt)
