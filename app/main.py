"""FastAPI application factory.

Run from the project root:
    uvicorn app.main:app --reload            # http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from config.settings import get_settings
from services.assistant import AssistantService

logger = logging.getLogger(__name__)


def create_app(service: AssistantService | None = None, warm_up: bool = True) -> FastAPI:
    """Build the app. Tests pass their own `service`; otherwise one is created from
    settings at startup (models are loaded then, not on the first request)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            settings = get_settings()
            app.state.service = AssistantService.from_settings(settings)
            if warm_up:
                logger.info("Loading models...")
                app.state.service.warm_up()
        else:
            app.state.service = service
        yield

    app = FastAPI(
        title="Enterprise AI Knowledge Assistant",
        description="Ask questions about your documents; answers cite their sources.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
