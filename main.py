import socket
import sys

import uvicorn
from loguru import logger

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


def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def resolve_port(settings: Settings, max_attempts: int = 100) -> int:
    if is_port_available(settings.api_host, settings.api_port):
        return settings.api_port

    if not settings.auto_port:
        logger.error(
            f"Port {settings.api_port} is already in use on {settings.api_host}. "
            "Stop the existing process or change API_PORT. "
            "Set AUTO_PORT=true to automatically select the next available port for local development."
        )
        raise SystemExit(1)

    for candidate_port in range(settings.api_port + 1, settings.api_port + max_attempts + 1):
        if is_port_available(settings.api_host, candidate_port):
            logger.warning(
                f"Port {settings.api_port} is already in use on {settings.api_host}; "
                f"using {candidate_port} because AUTO_PORT=true."
            )
            return candidate_port

    logger.error(
        f"Port {settings.api_port} is already in use and no available port was found "
        f"within the next {max_attempts} ports."
    )
    raise SystemExit(1)


def main():
    setup_logging()
    settings = Settings()
    port = resolve_port(settings)
    logger.info(f"Starting WQSurrogateModels API on {settings.api_host}:{port}")
    uvicorn.run(app, host=settings.api_host, port=port)

if __name__ == "__main__":
    main()
