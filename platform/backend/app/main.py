"""Local FastAPI service for the independent Retention Lab 2.0 platform."""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .core.runtime import Runtime
from .api.routes import router


def create_app(home: str | Path | None = None) -> FastAPI:
    platform_root = Path(__file__).resolve().parents[2]
    directory = Path(home or os.environ.get("RETENTION_PLATFORM_HOME", platform_root / "runtime"))
    rt = Runtime(directory)

    @asynccontextmanager
    async def lifespan(application):
        rt.store.recover_jobs()
        yield
        rt.shutdown()

    application = FastAPI(title="Retention Lab 2.0", version="2.0.0", lifespan=lifespan,
                          description="Local customer-level historical churn analysis. Predictions, associations and hypothetical economics are explicitly distinct.")
    application.state.runtime = rt
    application.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                               allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    application.include_router(router)

    @application.exception_handler(ValueError)
    async def invalid_value(request: Request, error: ValueError):
        return JSONResponse({"detail": str(error)}, status_code=422)

    @application.exception_handler(FileNotFoundError)
    async def missing_storage(request: Request, error: FileNotFoundError):
        logging.getLogger(__name__).exception("A local platform file was unavailable")
        return JSONResponse({"detail": "A stored dataset/model file is unavailable. Upload or train a new version."}, status_code=409)

    dist = platform_root / "frontend" / "dist"
    if (dist / "assets").is_dir():
        application.mount("/assets", StaticFiles(directory=dist / "assets"), name="frontend-assets")

    @application.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path.startswith("api/") or path in {"api", "assets"}:
            raise HTTPException(404, "Unknown API route.")
        if not (dist / "index.html").is_file():
            if path:
                raise HTTPException(404, "Build the React frontend or run Vite on port 5173.")
            return {"service": "Retention Lab 2.0 API", "docs": "/docs", "frontend": "Run Vite on http://127.0.0.1:5173, or build platform/frontend/dist."}
        return FileResponse(dist / "index.html")

    return application


app = create_app()
