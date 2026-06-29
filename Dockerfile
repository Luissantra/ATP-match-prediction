FROM python:3.11-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:$PORT app:app"]
