# Halyk Market — Image Search Backend v2

FastAPI-сервис: принимает изображение, вызывает Google Vision API, возвращает метки и русские ключевые слова.

## Локальный запуск

```bash
pip install -r requirements.txt
export GOOGLE_VISION_API_KEY=your_key_here   # Windows: set GOOGLE_VISION_API_KEY=...
uvicorn main:app --reload
```

Swagger UI: http://localhost:8000/docs

## Деплой на Render (бесплатно)

1. Аккаунт → https://render.com
2. New → Web Service → подключить репозиторий с папкой `halyk-backend/`
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Environment Variables: `GOOGLE_VISION_API_KEY=<ваш ключ>`
6. Deploy → скопировать URL вида `https://halyk-image-search.onrender.com`

> Render Free засыпает после 15 мин неактивности. Первый запрос после сна ~30 сек — норма для демо.

## Получить Google Vision API ключ

1. https://console.cloud.google.com → новый проект
2. APIs & Services → Enable APIs → "Cloud Vision API" → Enable
3. APIs & Services → Credentials → Create Credentials → API Key
4. Скопировать ключ

Первые 1 000 запросов/месяц бесплатно.

## Эндпоинты

`GET /` — health check, возвращает `{"status": "ok"}`

`POST /api/search/image` — multipart/form-data, поле `file` (jpg/png/webp, до 10 МБ)

```json
{
  "success": true,
  "labels_raw": ["sneakers", "footwear", "nike"],
  "labels_with_scores": [{"description": "sneakers", "score": 0.95, "source": "label"}],
  "keywords_ru": ["кроссовки", "обувь", "nike"]
}
```
