"""HTTP client the Streamlit UI uses to talk to the FastAPI backend.

Kept free of Streamlit and of the backend's own modules, so the UI can run on a
different machine from the API and this file can be tested on its own.
"""

from typing import Any

import requests

DEFAULT_TIMEOUT = 30          # seconds
ASK_TIMEOUT = 300             # local CPU models can take minutes
UPLOAD_TIMEOUT = 600          # large documents take a while to embed


class ApiError(Exception):
    """`status` is the HTTP status, or 0 when the API could not be reached."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class ApiClient:
    def __init__(self, base_url: str, session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, timeout: float = DEFAULT_TIMEOUT, **kwargs) -> Any:
        url = f"{self.base_url}/api{path}"
        try:
            response = self.session.request(method, url, timeout=timeout, **kwargs)
        except requests.ConnectionError as exc:
            raise ApiError(0, f"Cannot reach the API at {self.base_url}. Is it running?") from exc
        except requests.Timeout as exc:
            raise ApiError(0, "The API did not respond in time.") from exc
        if response.ok:
            return response.json()
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if isinstance(detail, list):  # FastAPI validation errors
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
        raise ApiError(response.status_code, str(detail))

    def health(self) -> dict:
        return self._request("GET", "/health", timeout=5)

    def documents(self) -> list[dict]:
        return self._request("GET", "/documents")

    def upload(self, filename: str, data: bytes, replace: bool = False) -> dict:
        return self._request("POST", "/documents", timeout=UPLOAD_TIMEOUT,
                             params={"replace": str(replace).lower()},
                             files={"file": (filename, data)})

    def delete(self, source: str) -> dict:
        return self._request("DELETE", f"/documents/{requests.utils.quote(source)}")

    def sync(self) -> dict:
        return self._request("POST", "/index/sync", timeout=UPLOAD_TIMEOUT)

    def search(self, query: str, top_k: int | None = None,
               sources: list[str] | None = None) -> dict:
        return self._request("POST", "/search",
                             json={"query": query, "top_k": top_k, "sources": sources or None})

    def ask(self, question: str, top_k: int | None = None,
            sources: list[str] | None = None) -> dict:
        return self._request("POST", "/ask", timeout=ASK_TIMEOUT,
                             json={"question": question, "top_k": top_k,
                                   "sources": sources or None})
