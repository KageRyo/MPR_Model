from loguru import logger
import sys

import uvicorn

from src.api import app
from src.settings import Settings


def setup_logging() -> None:
    """Configure loguru for nice console output."""
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )


def main():
    setup_logging()
    settings = Settings()
    logger.info(f"Starting WQSurrogateModels API on {settings.api_host}:{settings.api_port}")
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)

if __name__ == "__main__":
    main()
