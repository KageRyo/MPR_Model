FROM python:3.14-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir ".[models]"

EXPOSE 8001

CMD ["python", "main.py"]
