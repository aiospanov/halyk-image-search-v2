import os
import base64
import logging
from io import BytesIO

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Halyk Market Image Search API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")
VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"

# ──────────────────────────────────────────────────────────────
# LABEL_MAP — точный маппинг реальных меток Vision API → RU
# Метки взяты из реальных ответов Vision API для каждой категории
# ──────────────────────────────────────────────────────────────
LABEL_MAP = {

    # ── СМАРТФОНЫ ─────────────────────────────────────────────
    # Vision API для телефонов возвращает именно эти метки
    "mobile phone":                   ["смартфон", "телефон"],
    "smartphone":                     ["смартфон", "телефон"],
    "iphone":                         ["iphone", "apple", "смартфон"],
    "telephone":                      ["смартфон", "телефон"],
    "communication device":           ["смартфон", "телефон"],
    "portable communications device": ["смартфон", "телефон"],
    "gadget":                         ["смартфон", "телефон"],        # ← частая метка для любой электроники
    "electronic device":              ["смартфон", "телефон"],        # ← часто для телефонов
    "feature phone":                  ["смартфон", "телефон"],
    "handheld device":                ["смартфон", "планшет"],
    "phablet":                        ["смартфон"],
    "technology":                     ["смартфон", "ноутбук"],        # ← общая, но нужна
    "product":                        [],                             # ← слишком общая, игнорируем

    # ── НАУШНИКИ ──────────────────────────────────────────────
    "headphones":                     ["наушники"],
    "headset":                        ["наушники"],
    "earphones":                      ["наушники"],
    "earbuds":                        ["наушники"],
    "in-ear monitor":                 ["наушники"],
    "audio equipment":                ["наушники", "колонка"],
    "audio":                          ["наушники", "колонка"],
    "airpods":                        ["наушники", "airpods", "apple"],
    "wireless headphones":            ["наушники"],
    "noise-cancelling headphones":    ["наушники"],

    # ── КРОССОВКИ / ОБУВЬ ─────────────────────────────────────
    # Vision API для Nike/Adidas возвращает именно эти метки
    "sneakers":                       ["кроссовки"],
    "sneaker":                        ["кроссовки"],
    "athletic shoe":                  ["кроссовки"],
    "running shoe":                   ["кроссовки", "беговые"],
    "running shoes":                  ["кроссовки", "беговые"],
    "outdoor shoe":                   ["кроссовки"],                  # ← Nike часто получает это
    "walking shoe":                   ["кроссовки"],                  # ← Nike часто получает это
    "cross training shoe":            ["кроссовки"],
    "skate shoe":                     ["кроссовки", "кеды"],
    "basketball shoe":                ["кроссовки"],
    "footwear":                       ["кроссовки", "обувь"],
    "shoe":                           ["кроссовки", "обувь"],
    "boot":                           ["ботинки", "обувь"],
    "high heels":                     ["обувь"],
    "sandal":                         ["обувь"],
    "slipper":                        ["обувь"],
    # Бренды обуви как объекты
    "nike":                           ["nike", "кроссовки"],
    "adidas":                         ["adidas", "кроссовки"],
    "new balance":                    ["кроссовки", "new balance"],
    "puma":                           ["кроссовки", "puma"],
    "vans":                           ["кроссовки", "vans"],
    "converse":                       ["кроссовки", "кеды", "converse"],

    # ── НОУТБУКИ ──────────────────────────────────────────────
    "laptop":                         ["ноутбук"],
    "laptop computer":                ["ноутбук"],
    "notebook":                       ["ноутбук"],
    "netbook":                        ["ноутбук"],
    "computer":                       ["ноутбук", "компьютер"],
    "personal computer":              ["ноутбук", "компьютер"],
    "macbook":                        ["macbook", "ноутбук", "apple"],
    "ultrabook":                      ["ноутбук"],
    "chromebook":                     ["ноутбук"],

    # ── ТЕЛЕВИЗОРЫ ────────────────────────────────────────────
    "television":                     ["телевизор"],
    "television set":                 ["телевизор"],
    "tv":                             ["телевизор"],
    "flat panel display":             ["телевизор", "монитор"],
    "display device":                 ["монитор", "телевизор"],
    "led-backlit lcd display":        ["телевизор", "монитор"],
    "oled":                           ["телевизор"],

    # ── УМНЫЕ ЧАСЫ / ЧАСЫ ─────────────────────────────────────
    "smartwatch":                     ["часы", "умные часы"],
    "watch":                          ["часы"],
    "wristwatch":                     ["часы"],
    "analog watch":                   ["часы"],
    "digital watch":                  ["часы"],
    "apple watch":                    ["apple watch", "часы", "apple"],

    # ── ПЛАНШЕТЫ ──────────────────────────────────────────────
    "tablet computer":                ["планшет"],
    "tablet":                         ["планшет"],
    "ipad":                           ["ipad", "планшет", "apple"],

    # ── ИГРОВЫЕ КОНСОЛИ ───────────────────────────────────────
    "game controller":                ["консоль", "джойстик"],
    "gamepad":                        ["консоль", "джойстик"],
    "joystick":                       ["консоль", "джойстик"],
    "playstation":                    ["playstation", "консоль"],
    "xbox":                           ["xbox", "консоль"],
    "nintendo":                       ["nintendo", "консоль"],
    "video game console":             ["консоль"],

    # ── ФОТОАППАРАТЫ / КАМЕРЫ ─────────────────────────────────
    "camera":                         ["фотоаппарат", "камера"],
    "digital camera":                 ["фотоаппарат", "камера"],
    "mirrorless camera":              ["фотоаппарат"],
    "dslr camera":                    ["фотоаппарат"],
    "action camera":                  ["экшн-камера", "камера"],
    "video camera":                   ["камера"],
    "gopro":                          ["gopro", "камера"],
    "lens":                           ["фотоаппарат", "камера"],
    "single-lens reflex camera":      ["фотоаппарат"],

    # ── КОЛОНКИ ───────────────────────────────────────────────
    "loudspeaker":                    ["колонка"],
    "speaker":                        ["колонка"],
    "boombox":                        ["колонка"],
    "jbl":                            ["jbl", "колонка"],
    "marshall":                       ["marshall", "колонка"],

    # ── КОМПЬЮТЕРНЫЕ АКСЕССУАРЫ ───────────────────────────────
    "keyboard":                       ["клавиатура"],
    "computer keyboard":              ["клавиатура"],
    "mouse":                          ["мышь"],
    "computer mouse":                 ["мышь"],
    "monitor":                        ["монитор"],
    "computer monitor":               ["монитор"],

    # ── ОДЕЖДА ────────────────────────────────────────────────
    "jacket":                         ["куртка"],
    "outerwear":                      ["куртка", "одежда"],
    "coat":                           ["пальто", "куртка"],
    "hoodie":                         ["толстовка", "одежда"],
    "sweatshirt":                     ["толстовка", "одежда"],
    "t-shirt":                        ["футболка"],
    "jeans":                          ["джинсы"],
    "trousers":                       ["брюки", "одежда"],
    "shorts":                         ["шорты", "одежда"],
    "dress":                          ["платье", "одежда"],
    "clothing":                       ["одежда"],
    "sportswear":                     ["спортивная одежда"],
    "puffer jacket":                  ["пуховик", "куртка"],
    "down jacket":                    ["пуховик", "куртка"],

    # ── БЫТОВАЯ ТЕХНИКА ───────────────────────────────────────
    "vacuum cleaner":                 ["пылесос"],
    "robotic vacuum cleaner":         ["робот-пылесос"],
    "hair dryer":                     ["фен"],
    "hair iron":                      ["стайлер", "фен"],
    "kitchen appliance":              ["бытовая техника"],
    "coffee maker":                   ["кофемашина"],
    "espresso machine":               ["кофемашина"],
    "blender":                        ["блендер"],
    "microwave oven":                 ["микроволновка"],
    "washing machine":                ["стиральная машина"],

    # ── ДРОНЫ ─────────────────────────────────────────────────
    "drone":                          ["дрон"],
    "quadcopter":                     ["дрон"],
    "unmanned aerial vehicle":        ["дрон"],
    "dji":                            ["dji", "дрон"],

    # ── АКСЕССУАРЫ ────────────────────────────────────────────
    "sunglasses":                     ["очки", "солнцезащитные"],
    "glasses":                        ["очки"],
    "backpack":                       ["рюкзак"],
    "bag":                            ["сумка"],
    "suitcase":                       ["чемодан"],
    "luggage":                        ["чемодан", "багаж"],

    # ── ОБЩИЕ (намеренно пустые или широкие) ──────────────────
    "product":                        [],   # слишком общая
    "brand":                          [],   # слишком общая
    "fashion":                        ["одежда", "обувь"],
    "electronics":                    ["смартфон", "ноутбук"],
    "fashion accessory":              ["аксессуары", "часы"],
}

# Метки которые Vision API возвращает часто но они бесполезны для поиска
IGNORE_LABELS = {
    "product", "brand", "font", "logo", "icon", "image", "photo",
    "photography", "still life", "close-up", "white", "black", "color",
    "design", "pattern", "material", "texture", "art", "illustration",
    "rectangle", "square", "circle", "shape", "line",
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
        raise HTTPException(status_code=500, detail="GOOGLE_VISION_API_KEY not configured")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [{
            "image": {"content": b64},
            "features": [
                {"type": "LABEL_DETECTION",       "maxResults": 15},
                {"type": "LOGO_DETECTION",         "maxResults": 5},
                {"type": "OBJECT_LOCALIZATION",    "maxResults": 10},
            ]
        }]
    }

    async with httpx.AsyncClient(timeout=12.0) as client:
        resp = await client.post(
            f"{VISION_API_URL}?key={GOOGLE_VISION_API_KEY}",
            json=payload
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Vision API error: {resp.text}")
        return resp.json()


def parse_vision_response(vision_data: dict) -> list[dict]:
    result = vision_data.get("responses", [{}])[0]
    labels = []

    # LABEL_DETECTION — основные метки (порог снижен до 0.60)
    for item in result.get("labelAnnotations", []):
        desc = item["description"].lower()
        score = item.get("score", 0)
        if score >= 0.60 and desc not in IGNORE_LABELS:
            labels.append({"description": desc, "score": round(score, 2), "source": "label"})

    # LOGO_DETECTION — бренды (Nike, Apple, Samsung...) — всегда добавляем
    for item in result.get("logoAnnotations", []):
        desc = item["description"].lower()
        labels.append({"description": desc, "score": 0.95, "source": "logo"})

    # OBJECT_LOCALIZATION — локализованные объекты на фото
    for item in result.get("localizedObjectAnnotations", []):
        desc = item["name"].lower()
        score = item.get("score", 0)
        if score >= 0.55 and desc not in IGNORE_LABELS:
            labels.append({"description": desc, "score": round(score, 2), "source": "object"})

    logger.info(f"Vision API returned {len(labels)} labels: {[l['description'] for l in labels]}")
    return labels


def labels_to_ru_keywords(labels: list[dict]) -> list[str]:
    keywords = set()
    unmapped = []

    for label in labels:
        desc = label["description"].lower()

        # 1. Точное совпадение в словаре
        if desc in LABEL_MAP:
            mapped = LABEL_MAP[desc]
            if mapped:  # пустой список = намеренно игнорируем
                keywords.update(mapped)
            continue

        # 2. Частичное совпадение — ищем только если метка длиннее 4 символов
        # и только совпадение НАЧАЛА ключа (избегаем ложных срабатываний)
        matched = False
        if len(desc) > 4:
            for key, values in LABEL_MAP.items():
                if len(key) > 4 and (desc.startswith(key) or key.startswith(desc)):
                    if values:
                        keywords.update(values)
                    matched = True
                    break

        if not matched:
            unmapped.append(desc)

    logger.info(f"Mapped keywords: {list(keywords)}")
    logger.info(f"Unmapped labels (no dict entry): {unmapped}")

    return list(keywords)


@app.get("/")
async def health():
    return {"status": "ok", "service": "Halyk Market Image Search API", "version": "1.1.0"}


@app.post("/api/search/image")
async def search_by_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Файл слишком большой (макс. 10 МБ)")

    compressed = compress_image(raw)
    vision_data = await call_vision_api(compressed)
    labels = parse_vision_response(vision_data)
    ru_keywords = labels_to_ru_keywords(labels)

    return JSONResponse({
        "success": True,
        "labels_raw": [l["description"] for l in labels],       # все метки для отладки
        "labels_with_scores": labels,
        "keywords_ru": ru_keywords,
        "unmapped_labels": [                                     # метки без маппинга — для пополнения словаря
            l["description"] for l in labels
            if l["description"] not in LABEL_MAP
            and not any(l["description"].startswith(k) or k.startswith(l["description"])
                       for k in LABEL_MAP if len(k) > 4)
        ]
    })
