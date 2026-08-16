"""
Structured logging and Prometheus-style metrics.

Drift in the prediction distribution is the earliest signal that something has
gone wrong in production, so predictions are counted by class alongside the
usual latency and error counters.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections import defaultdict
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("brainscan")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.handlers = [handler]
    logger.setLevel(level.upper())
    logger.propagate = False


def log_event(message: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, message, extra={"extra_fields": fields})


class Metrics:
    """Minimal in-process counters rendered in Prometheus text format."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.requests: dict[tuple[str, int], int] = defaultdict(int)
        self.latency_sum: dict[str, float] = defaultdict(float)
        self.latency_count: dict[str, int] = defaultdict(int)
        self.predictions: dict[str, int] = defaultdict(int)
        self.ood_rejections: int = 0
        self.flagged_for_review: int = 0
        self.errors: dict[str, int] = defaultdict(int)

    def observe_request(self, path: str, status_code: int, duration: float) -> None:
        with self._lock:
            self.requests[(path, status_code)] += 1
            self.latency_sum[path] += duration
            self.latency_count[path] += 1
            if status_code >= 500:
                self.errors[path] += 1

    def observe_prediction(self, predicted_class: str, flagged: bool, is_ood: bool) -> None:
        with self._lock:
            self.predictions[predicted_class] += 1
            if flagged:
                self.flagged_for_review += 1
            if is_ood:
                self.ood_rejections += 1

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            lines.append("# HELP brainscan_requests_total Total HTTP requests by path and status.")
            lines.append("# TYPE brainscan_requests_total counter")
            for (path, status), count in sorted(self.requests.items()):
                lines.append(f'brainscan_requests_total{{path="{path}",status="{status}"}} {count}')

            lines.append("# HELP brainscan_request_duration_seconds Cumulative request duration.")
            lines.append("# TYPE brainscan_request_duration_seconds summary")
            for path, total in sorted(self.latency_sum.items()):
                lines.append(f'brainscan_request_duration_seconds_sum{{path="{path}"}} {total:.6f}')
                lines.append(
                    f'brainscan_request_duration_seconds_count{{path="{path}"}} {self.latency_count[path]}'
                )

            lines.append("# HELP brainscan_predictions_total Predictions by class (watch for drift).")
            lines.append("# TYPE brainscan_predictions_total counter")
            for cls, count in sorted(self.predictions.items()):
                lines.append(f'brainscan_predictions_total{{class="{cls}"}} {count}')

            lines.append("# HELP brainscan_ood_rejections_total Inputs rejected as out of distribution.")
            lines.append("# TYPE brainscan_ood_rejections_total counter")
            lines.append(f"brainscan_ood_rejections_total {self.ood_rejections}")

            lines.append("# HELP brainscan_flagged_total Results flagged for specialist review.")
            lines.append("# TYPE brainscan_flagged_total counter")
            lines.append(f"brainscan_flagged_total {self.flagged_for_review}")

            lines.append("# HELP brainscan_errors_total Server errors by path.")
            lines.append("# TYPE brainscan_errors_total counter")
            for path, count in sorted(self.errors.items()):
                lines.append(f'brainscan_errors_total{{path="{path}"}} {count}')

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()
            self.latency_sum.clear()
            self.latency_count.clear()
            self.predictions.clear()
            self.errors.clear()
            self.ood_rejections = 0
            self.flagged_for_review = 0


metrics = Metrics()


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Attach a request ID, emit a structured access log, and record metrics."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - started
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            metrics.observe_request(path, 500, duration)
            log_event(
                "request_failed", level=logging.ERROR,
                request_id=request_id, method=request.method, path=path,
                duration_ms=round(duration * 1000, 2),
            )
            raise

        duration = time.perf_counter() - started
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        metrics.observe_request(path, response.status_code, duration)
        response.headers["X-Request-ID"] = request_id

        log_event(
            "request",
            request_id=request_id,
            method=request.method,
            path=path,
            status=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )
        return response
