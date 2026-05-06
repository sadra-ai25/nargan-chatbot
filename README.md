# سیستم RAG مستندات نارگان

سیستم بازیابی و پاسخگویی هوشمند مستندات و اکسل مهندسی نارگان

## 🚀 شروع سریع

### روش ۱: با Docker

```bash
# ساخت و اجرا
docker-compose up --build

# اجرا در پس‌زمینه
docker-compose up -d

# Fast RUN:

uvicorn api.main:app --host 0.0.0.0 --port 9005 --reload