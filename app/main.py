"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import health, inpaint, presets, process, segment
from app.config import get_settings
from app.exceptions import AppError
from app.monitoring import metrics_response, start_metrics_server


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks."""
    settings = get_settings()
    if settings.enable_metrics:
        start_metrics_server(settings.metrics_port)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Exception handlers — return JSON instead of HTML
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.__class__.__name__, "message": exc.message},
        )

    # API routes
    app.include_router(health.router)
    app.include_router(presets.router)
    app.include_router(process.router)
    app.include_router(inpaint.router)
    app.include_router(segment.router)

    # Metrics
    @app.get("/metrics")
    def metrics():
        body, content_type = metrics_response()
        return JSONResponse(content=body.decode("utf-8"), media_type=content_type)

    # Static files (web/) at root
    web_dir = Path(__file__).parent.parent / "web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1,
    )
