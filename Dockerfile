# FROM python:3.10-slim

# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements first for caching
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . .

# # Create data directory
# RUN mkdir -p /app/data

# # Expose port
# EXPOSE 9005

# # Default command
# CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "9005"]
# استفاده از نسخه کامل به جای slim (این نسخه build-essential را دارد)



# FROM python:3.10

# WORKDIR /app

# # مرحله نصب پیش‌نیازهای سیستم حذف شد چون در ایمیج اصلی موجود است
# # اگر پکیج خیلی خاصی نیاز ندارید، این ۳ خط را حذف کنید:
# # RUN apt-get update && apt-get install -y \
# #    build-essential \
# #    && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .
# RUN mkdir -p /app/data
# EXPOSE 9005
# CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "9005"]


FROM python:3.10-slim

WORKDIR /app

# System dependencies for PDF and FAISS
# RUN apt-get update && apt-get install -y \
#     libgomp1 \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/documents/raw /app/data/excel /app/data/documents/faiss_index

# Pre-download E5 model (optional, if not mounted)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"

EXPOSE 9005

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "9005"]