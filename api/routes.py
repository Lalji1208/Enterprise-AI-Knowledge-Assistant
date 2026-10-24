"""REST endpoints. All business logic lives in services.assistant.AssistantService.

Routes are plain `def` (not `async def`): the work is CPU-bound model inference
and file I/O, so FastAPI runs each request in its thread pool.
"""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from api.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    SearchResponse,
    SyncResponse,
)
from rag.llm import LLMError
from rag.retriever import RetrievalError
from services.assistant import (
    AssistantService,
    DocumentNotFoundError,
    UploadError,
    report_to_dict,
)

router = APIRouter(prefix="/api")
ERRORS = {code: {"model": ErrorResponse} for code in (400, 404, 409, 413, 415, 422, 503)}
UPLOAD_READ_BLOCK = 1024 * 1024


def service(request: Request) -> AssistantService:
    return request.app.state.service


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> dict:
    return service(request).health()


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(request: Request) -> list[dict]:
    return service(request).list_documents()


@router.post("/documents", response_model=SyncResponse, responses=ERRORS)
def upload_document(
    request: Request,
    file: Annotated[UploadFile, File()],
    replace: Annotated[bool, Query(description="overwrite a file with the same name")] = False,
) -> dict:
    svc = service(request)
    # Read in blocks so an oversized upload is rejected without loading all of it.
    limit = svc.settings.max_upload_size_mb * 1024 * 1024
    data = bytearray()
    while block := file.file.read(UPLOAD_READ_BLOCK):
        data.extend(block)
        if len(data) > limit:
            raise HTTPException(413, f"File is larger than {svc.settings.max_upload_size_mb} MB.")
    try:
        return report_to_dict(svc.upload(file.filename or "", bytes(data), replace=replace))
    except UploadError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.delete("/documents/{source}", response_model=SyncResponse, responses=ERRORS)
def delete_document(source: str, request: Request) -> dict:
    try:
        return report_to_dict(service(request).delete(source))
    except UploadError as exc:
        raise HTTPException(400, str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(404, f"No document named '{exc}'.") from exc


@router.post("/index/sync", response_model=SyncResponse)
def sync_index(request: Request) -> dict:
    """Re-scan DOCUMENTS_DIR (e.g. after copying files in manually)."""
    return report_to_dict(service(request).sync())


@router.post("/search", response_model=SearchResponse, responses=ERRORS)
def search(body: QueryRequest, request: Request) -> dict:
    try:
        results = service(request).search(body.query, top_k=body.top_k, sources=body.sources)
    except RetrievalError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"query": body.query, "results": results}


@router.post("/ask", response_model=AskResponse, responses=ERRORS)
def ask(body: AskRequest, request: Request):
    try:
        return service(request).ask(body.question, top_k=body.top_k, sources=body.sources)
    except RetrievalError as exc:
        raise HTTPException(400, str(exc)) from exc
    except LLMError as exc:
        # The service is up but its LLM dependency is not usable right now.
        raise HTTPException(503, str(exc)) from exc
