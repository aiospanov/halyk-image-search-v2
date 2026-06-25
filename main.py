import os
import base64
import json
import logging
import re
from io import BytesIO

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Halyk Market Image Search API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

SYSTEM_PROMPT = """Ты — система анализа товаров для казахстанского маркетплейса Halyk Market.

Посмотри на фото и определи товар. Верни ТОЛЬКО JSON без лишнего текста.

Правила:
- brand: название бренда если виден (Nike, Apple, Samsung, Adidas и т.д.) или null
- model: конкретная модель если можешь определить (Air Max 270, iPhone 15 и т.д.) или null  
- category: одна из категорий: смартфон, наушники, кроссовки, ноутбук, телевизор, планшет, умные часы, игровая консоль, фотоаппарат, одежда, пылесос, кофемашина, колонка, дрон, монитор, другое
- keywords: список из 3-6 ключевых слов на русском для поиска товара в магазине
- confidence: твоя уверенность от 0.0 до 1.0

Пример ответа:
{"brand": "Nike", "model": "Air Max 270", "category": "кроссовки", "keywords": ["кроссовки", "nike", "air max", "беговые"], "confidence": 0.92}

Если на фото нет товара или непонятно что изображено:
{"brand": null, "model": null, "category": "другое", "keywords": [], "confidence": 0.1}"""


def compress_image(image_bytes: bytes, max_size_kb: int = 800) -> bytes:
    img = Image.open(BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    output = BytesIO()
    quality = 85
    img.save(output, format="JPEG", quality=quality, optimize=True)
    while output.tell() > max_size_kb * 1024 and quality > 40:
        quality -= 10
        output = BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


async def call_gemini(image_bytes: bytes) -> dict:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY не настроен")

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                {"text": SYSTEM_PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 512,
            # responseMimeType убран — не все версии модели его поддерживают
        }
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        logger.info(f"Gemini HTTP status: {resp.status_code}")
        if resp.status_code != 200:
            logger.error(f"Gemini error body: {resp.text[:500]}")
            raise HTTPException(status_code=502, detail=f"Gemini API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        logger.info(f"Gemini raw response: {json.dumps(data)[:800]}")
        return data


def parse_gemini_response(raw: dict) -> dict:
    try:
        # Логируем полную структуру для отладки
        logger.info(f"Parsing response keys: {list(raw.keys())}")

        candidates = raw.get("candidates", [])
        if not candidates:
            logger.error(f"No candidates in response. Full raw: {raw}")
            return {"brand": None, "model": None, "category": "другое", "keywords": [], "confidence": 0.0}

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        logger.info(f"Finish reason: {finish_reason}")

        # Проверяем блокировку по safety
        if finish_reason == "SAFETY":
            logger.warning("Gemini blocked response due to safety filters")
            return {"brand": None, "model": None, "category": "другое", "keywords": [], "confidence": 0.0}

        content = candidate.get("content", {})
        parts = content.get("parts", [])
        if not parts:
            logger.error(f"No parts in content: {content}")
            return {"brand": None, "model": None, "category": "другое", "keywords": [], "confidence": 0.0}

        text = parts[0].get("text", "")
        logger.info(f"Gemini text response: {repr(text[:300])}")

        # Убираем markdown-обёртку которую Gemini иногда добавляет
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        # Ищем JSON в тексте если Gemini добавил лишний текст вокруг
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        result = json.loads(text)
        logger.info(f"Parsed result: {result}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}. Text was: {repr(text[:300]) if 'text' in dir() else 'N/A'}")
        return {"brand": None, "model": None, "category": "другое", "keywords": [], "confidence": 0.0}
    except Exception as e:
        logger.error(f"Unexpected error parsing Gemini response: {e}. Raw: {str(raw)[:300]}")
        return {"brand": None, "model": None, "category": "другое", "keywords": [], "confidence": 0.0}


def build_search_labels(result: dict) -> list[str]:
    """Формируем читаемые метки для отображения на фронтенде."""
    labels = []
    if result.get("brand"):
        labels.append(result["brand"])
    if result.get("model"):
        labels.append(result["model"])
    if result.get("category") and result["category"] != "другое":
        labels.append(result["category"])
    return labels



@app.post("/api/debug")
async def debug_gemini(file: UploadFile = File(...)):
    """Сырой ответ Gemini для отладки — удали в production."""
    raw_bytes = await file.read()
    compressed = compress_image(raw_bytes)
    gemini_raw = await call_gemini(compressed)
    return JSONResponse(gemini_raw)

@app.get("/")
async def health():
    return {
        "status": "ok",
        "service": "Halyk Market Image Search API",
        "version": "2.0.0",
        "engine": "Google Gemini 2.5 Flash"
    }


@app.post("/api/search/image")
async def search_by_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Файл слишком большой (макс. 10 МБ)")

    compressed = compress_image(raw)
    gemini_raw  = await call_gemini(compressed)
    result      = parse_gemini_response(gemini_raw)

    return JSONResponse({
        "success":     True,
        "brand":       result.get("brand"),
        "model":       result.get("model"),
        "category":    result.get("category"),
        "keywords_ru": result.get("keywords", []),
        "confidence":  result.get("confidence", 0),
        "labels_raw":  build_search_labels(result),
    })
