import uvicorn

from src.api import app
from src.settings import Settings

def main():
    settings = Settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)

if __name__ == "__main__":
    main()
