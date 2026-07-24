"""
Minimal Python client for the s3-gateway REST API.

Requires: requests  (pip install requests)

Auth: HTTP Basic + optional X-Tenant-Id header. Pass tenant=None on reads to resolve
by id across tenants (reader role). Writes require a tenant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, BinaryIO
import requests
import threading

import config

import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Client class and helpers
# --------------------------------------------------------------------------- #
# The S3GatewayClient is a thin wrapper around the s3-gateway REST API. It handles
# auth, tenant headers, and error checking. 
class S3GatewayError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


@dataclass
class DownloadHeaders:
    filename: Optional[str]
    content_type: str
    content_length: int


class S3GatewayClient:
    def __init__(self, base_url: str, username: str, password: str,
                 default_tenant: Optional[str] = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password)
        self.default_tenant = default_tenant
        self.timeout = timeout

    # ----- helpers -----
    def _headers(self, tenant: Optional[str]) -> dict:
        return {"X-Tenant-Id": tenant} if tenant else {}

    def _require_tenant(self, tenant: Optional[str]) -> str:
        t = tenant or self.default_tenant
        if not t:
            raise ValueError("A tenant is required for this operation")
        return t

    def _check(self, resp: requests.Response) -> None:
        if resp.status_code >= 400:
            raise S3GatewayError(resp.status_code, resp.text[:200])

    # ----- writes -----
    def upload(self, tenant: Optional[str], content: bytes, filename: str,
               content_type: str = "application/octet-stream") -> dict:
        resp = requests.post(
            f"{self.base_url}/api/v1/files",
            auth=self.auth,
            headers=self._headers(self._require_tenant(tenant)),
            files={"file": (filename, content, content_type)},
            timeout=self.timeout,
        )
        self._check(resp)
        return resp.json()

    def update(self, tenant: Optional[str], file_id: str, content: bytes, filename: str,
               content_type: str = "application/octet-stream") -> dict:
        resp = requests.put(
            f"{self.base_url}/api/v1/files/{file_id}",
            auth=self.auth,
            headers=self._headers(self._require_tenant(tenant)),
            files={"file": (filename, content, content_type)},
            timeout=self.timeout,
        )
        self._check(resp)
        return resp.json()

    def delete(self, tenant: Optional[str], file_id: str) -> None:
        resp = requests.delete(
            f"{self.base_url}/api/v1/files/{file_id}",
            auth=self.auth,
            headers=self._headers(self._require_tenant(tenant)),
            timeout=self.timeout,
        )
        self._check(resp)

    # ----- reads (tenant optional) -----
    def list(self, tenant: Optional[str] = None, page: int = 0, size: int = 20) -> dict:
        resp = requests.get(
            f"{self.base_url}/api/v1/files",
            auth=self.auth,
            headers=self._headers(tenant),
            params={"page": page, "size": size},
            timeout=self.timeout,
        )
        self._check(resp)
        return resp.json()

    def get_metadata(self, file_id: str, tenant: Optional[str] = None) -> dict:
        resp = requests.get(
            f"{self.base_url}/api/v1/files/{file_id}",
            auth=self.auth,
            headers=self._headers(tenant),
            timeout=self.timeout,
        )
        self._check(resp)
        return resp.json()

    def download_stream(self, file_id: str, tenant: Optional[str] = None,
                        chunk_size: int = 64 * 1024):
        """
        Returns (DownloadHeaders, generator). Use stream=True so bytes are not buffered.
        IMPORTANT: fully consume the generator (it holds the open connection).
        """
        resp = requests.get(
            f"{self.base_url}/api/v1/files/{file_id}/download",
            auth=self.auth,
            headers=self._headers(tenant),
            stream=True,
            timeout=self.timeout,
        )
        self._check(resp)

        filename = None
        cd = resp.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            filename = cd.split("filename=", 1)[1].strip().strip('"')
        headers = DownloadHeaders(
            filename=filename,
            content_type=resp.headers.get("Content-Type", "application/octet-stream"),
            content_length=int(resp.headers.get("Content-Length", -1)),
        )

        def gen() -> Iterator[bytes]:
            try:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        yield chunk
            finally:
                resp.close()

        return headers, gen()

    def download_to(self, file_id: str, sink: BinaryIO, tenant: Optional[str] = None) -> DownloadHeaders:
        headers, gen = self.download_stream(file_id, tenant)
        for chunk in gen:
            sink.write(chunk)
        return headers


# --------------------------------------------------------------------------- #
# Shared client instance
# --------------------------------------------------------------------------- #
# Module-level singleton, mirroring milvus.get_milvus_client(). The client holds
# credentials and config only (no connection state), so one instance is safely
# shared across threads.
_client: Optional[S3GatewayClient] = None
_client_lock = threading.Lock()


def get_csx_storage_client() -> S3GatewayClient:
    """Return the shared CSX storage client, creating it on first call."""
    global _client
    with _client_lock:
        if _client is None:
            logger.info(
                "Initializing CSX storage client for %s (tenant=%s)",
                config.CSX_BASE_URL, config.CSX_TENANT or "none",
            )
            _client = S3GatewayClient(
                base_url=config.CSX_BASE_URL,
                username=config.CSX_USERNAME,
                password=config.CSX_PASSWORD,
                default_tenant=config.CSX_TENANT,
                timeout=config.CSX_TIMEOUT,
            )
        return _client
    