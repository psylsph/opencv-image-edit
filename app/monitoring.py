"""Prometheus metrics for the image editor.

Defines the histograms and counters used by the FastAPI process endpoint
and the ``/metrics`` HTTP endpoint. The metrics HTTP server is started
in the FastAPI ``lifespan`` hook when ``settings.enable_metrics`` is True.
"""
from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from prometheus_client import start_http_server as _start_http_server


# Histograms
image_process_seconds = Histogram(
    "image_process_seconds",
    "Time spent processing an image end-to-end",
    labelnames=["status"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
request_size_bytes = Histogram(
    "request_size_bytes",
    "Size of incoming image requests in bytes",
    buckets=(1024, 100 * 1024, 500 * 1024, 1024 * 1024, 5 * 1024 * 1024, 10 * 1024 * 1024),
)

# Counters
image_process_total = Counter(
    "image_process_total",
    "Total number of image processing requests",
    labelnames=["status"],
)
model_cache_hits_total = Counter(
    "model_cache_hits_total",
    "Number of times a cached model was used",
    labelnames=["model"],
)
errors_total = Counter(
    "errors_total",
    "Number of errors",
    labelnames=["type"],
)


def start_metrics_server(port: int) -> None:
    """Start the Prometheus metrics HTTP server on the given port."""
    try:
        _start_http_server(port)
    except OSError as exc:
        # Port already in use — log and continue
        print(f"WARNING: metrics server could not bind port {port}: {exc}")


def metrics_response() -> tuple[bytes, str]:
    """Return (body, content_type) for /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
