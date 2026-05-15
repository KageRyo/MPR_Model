FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir ".[models]"

EXPOSE 8001

CMD ["python", "main.py"]
