FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements-railway.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-railway.txt

COPY . .

EXPOSE 8501

CMD ["python", "app.py", "--host", "0.0.0.0"]
