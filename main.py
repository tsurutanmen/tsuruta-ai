import os
import time
import hashlib
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# ==============================
# 🔑 環境変数（Renderで設定）
# ==============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
DEFAULT_AI_NAME = os.getenv("AI_NAME", "TSURUKAME CORE")

if not GEMINI_API_KEY:
    # 起動はできるけど、呼び出し時に分かりやすくエラーにする
    print("WARNING: GEMINI_API_KEY is not set.")

# ==============================
# 📦 リクエストモデル
# ==============================
class AIRequest(BaseModel):
    text: str
    user_id: str | None = None
    ai_name: str | None = None  # ユーザー別に上書き可能

# ==============================
# 💾 かんたんキャッシュ
# ==============================
CACHE: dict[str, tuple[float, dict]] = {}
CACHE_TTL = 60  # 秒

def _cache_key(prompt: str) -> str:
    return hashlib.sha256((GEMINI_MODEL + ":" + prompt).encode("utf-8")).hexdigest()

# ==============================
# 🌌 Gemini 呼び出し
# ==============================
def call_gemini(prompt: str):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail={"message": "GEMINI_API_KEY が未設定です（RenderのEnvに設定して）"})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    ck = _cache_key(prompt)
    now = time.time()
    if ck in CACHE:
        t, v = CACHE[ck]
        if now - t < CACHE_TTL:
            return v

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 256,  # 節約
            "temperature": 0.7
        }
    }

    r = requests.post(url, json=payload, timeout=30)

    # 成功
    if r.status_code == 200:
        js = r.json()
        CACHE[ck] = (time.time(), js)
        return js

    # 429（上限/混雑）
    if r.status_code == 429:
        try:
            js = r.json()
        except Exception:
            js = {"error": "rate_limited"}

        raise HTTPException(
            status_code=503,
            detail={
                "message": "Geminiが混雑/上限です。少し待ってから再試行してください。",
                "gemini": js
            }
        )

    # その他エラー
    try:
        return r.json()
    except Exception:
        raise HTTPException(status_code=500, detail={"message": "Gemini呼び出し失敗", "raw": r.text})

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
    system_prompt = f"""
あなたは「{ai_name}」。
ユーザーの思考を拡張し、未来を示すAI。

ユーザーID: {req.user_id}

ユーザー入力:
{req.text}
"""
    return call_gemini(system_prompt)

# 👤 プロファイル推定AI（年齢・性別）
@app.post("/profile")
def profile(req: AIRequest):
    prompt = f"""
次の文章から年齢層と性別を推定し、JSONだけで返せ。
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
    prompt = f"""
あなたは「{ai_name}」として振る舞うユーザーの分身AI。
次の問いに対して最適な判断を短く出せ。

ユーザーID: {req.user_id}

問い:
{req.text}
"""
    return call_gemini(prompt)
