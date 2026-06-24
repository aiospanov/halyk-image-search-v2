import os
import base64
from io import BytesIO

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI(title="Halyk Market Image Search API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")
VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"

LABEL_MAP = {
    "mobile phone": ["смартфон", "телефон"],
    "smartphone": ["смартфон", "телефон"],
    "iphone": ["iphone", "айфон", "apple"],
    "telephone": ["телефон", "смартфон"],
    "communication device": ["смартфон", "телефон"],
    "portable communications device": ["смартфон", "телефон"],
    "headphones": ["наушники"],
    "earphones": ["наушники", "tws"],
    "audio equipment": ["наушники", "колонка", "аудио"],
    "earbuds": ["наушники", "tws", "беспроводные"],
    "sneakers": ["кроссовки", "кеды"],
    "running shoes": ["кроссовки", "беговые"],
    "footwear": ["обувь", "кроссовки"],
    "shoe": ["обувь", "кроссовки"],
    "athletic shoe": ["кроссовки", "спортивная обувь"],
    "boot": ["ботинки", "обувь"],
    "laptop": ["ноутбук"],
    "computer": ["ноутбук", "компьютер"],
    "personal computer": ["ноутбук", "компьютер"],
    "netbook": ["ноутбук"],
    "television": ["телевизор"],
    "tv": ["телевизор"],
    "flat panel display": ["телевизор", "монитор"],
    "display device": ["монитор", "телевизор"],
    "smartwatch": ["часы", "умные часы"],
    "watch": ["часы"],
    "wristwatch": ["часы"],
    "tablet computer": ["планшет"],
    "tablet": ["планшет"],
    "ipad": ["ipad", "планшет"],
    "gaming": ["игровой", "консоль"],
    "game controller": ["консоль", "джойстик", "игровой"],
    "joystick": ["джойстик", "консоль"],
    "coffee maker": ["кофемашина", "кофе"],
    "espresso machine": ["кофемашина", "эспрессо"],
    "jacket": ["куртка", "пуховик"],
    "outerwear": ["куртка", "одежда"],
    "coat": ["пальто", "куртка"],
    "t-shirt": ["футболка", "одежда"],
    "jeans": ["джинсы", "одежда"],
    "camera": ["фотоаппарат", "камера"],
    "digital camera": ["фотоаппарат", "камера"],
    "speaker": ["колонка", "аудио"],
    "loudspeaker": ["колонка", "аудио"],
    "keyboard": ["клавиатура"],
    "computer keyboard": ["клавиатура"],
    "mouse": ["мышь"],
    "computer mouse": ["мышь"],
    "monitor": ["монитор"],
    "drone": ["дрон", "квадрокоптер"],
    "quadcopter": ["дрон", "квадрокоптер"],
    "vacuum cleaner": ["пылесос"],
    "kitchen appliance": ["техника", "кухня"],
    "hair dryer": ["фен", "стайлер"],
    "glasses": ["очки"],
    "sunglasses": ["очки", "солнцезащитные"],
    "luggage": ["чемодан", "багаж"],
    "suitcase": ["чемодан"],
    "book": ["книга"],
    "toy": ["игрушки"],
    "backpack": ["рюкзак"],
    # logo mappings
    "nike": ["nike", "кроссовки"],
    "apple": ["apple", "iphone", "ipad"],
    "samsung": ["samsung", "смартфон"],
    "sony": ["sony", "наушники"],
    "adidas": ["adidas", "кроссовки"],
    "xiaomi": ["xiaomi", "смартфон"],
    "lg": ["lg", "телевизор"],
}


def compress_image(image_bytes: bytes, max_size_kb: int = 800) -> bytes:
    img = Image.open(BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail((1200, 1200), Image.LANCZOS)
    output = BytesIO()
    quality = 85
    img.save(output, format="JPEG", quality=quality, optimize=True)
    while output.tell() > max_size_kb * 1024 and quality > 40:
        quality -= 10
        output = BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


async def call_vision_api(image_bytes: bytes) -> dict:
    if not GOOGLE_VISION_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_VISION_API_KEY не настроен")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [{
            "image": {"content": b64},
            "features": [
                {"type": "LABEL_DETECTION", "maxResults": 10},
                {"type": "LOGO_DETECTION", "maxResults": 3},
                {"type": "OBJECT_LOCALIZATION", "maxResults": 5},
            ],
        }]
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{VISION_API_URL}?key={GOOGLE_VISION_API_KEY}",
            json=payload,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Vision API error: {resp.text}")
        return resp.json()


def parse_vision_response(vision_data: dict) -> list[dict]:
    result = vision_data.get("responses", [{}])[0]
    labels = []

    for item in result.get("labelAnnotations", []):
        if item.get("score", 0) >= 0.65:
            labels.append({
                "description": item["description"].lower(),
                "score": round(item["score"], 2),
                "source": "label",
            })

    for item in result.get("logoAnnotations", []):
        labels.append({
            "description": item["description"].lower(),
            "score": 0.95,
            "source": "logo",
        })

    for item in result.get("localizedObjectAnnotations", []):
        if item.get("score", 0) >= 0.60:
            labels.append({
                "description": item["name"].lower(),
                "score": round(item["score"], 2),
                "source": "object",
            })

    return labels[:12]


def labels_to_ru_keywords(labels: list[dict]) -> list[str]:
    keywords: set[str] = set()
    for label in labels:
        desc = label["description"].lower()
        if desc in LABEL_MAP:
            keywords.update(LABEL_MAP[desc])
        else:
            for key, values in LABEL_MAP.items():
                if key in desc or desc in key:
                    keywords.update(values)
                    break
            else:
                keywords.add(desc)
    return list(keywords)


@app.get("/")
async def health():
    return {"status": "ok", "service": "Halyk Market Image Search API", "version": "2.0.0"}


@app.post("/api/search/image")
async def search_by_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Файл слишком большой (максимум 10 МБ)")

    compressed = compress_image(raw)
    vision_data = await call_vision_api(compressed)
    labels = parse_vision_response(vision_data)
    ru_keywords = labels_to_ru_keywords(labels)

    return JSONResponse({
        "success": True,
        "labels_raw": [lb["description"] for lb in labels[:5]],
        "labels_with_scores": labels[:5],
        "keywords_ru": ru_keywords,
    })
