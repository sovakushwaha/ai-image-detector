from __future__ import annotations

import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import ALLOWED_EXT, ALLOWED_MIME, BRAND_NAME, BRAND_SUBTITLE, MAX_UPLOAD_BYTES
from .inference import EnsembleService, open_image_bytes

_service: EnsembleService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = EnsembleService.create()
    yield
    _service = None


app = FastAPI(
    title=f"{BRAND_NAME} API",
    description=f"{BRAND_SUBTITLE} — local H4 research prototype demo",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def service() -> EnsembleService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Backend unavailable")
    return _service


@app.get("/api/health")
def health() -> dict[str, Any]:
    svc = service()
    return {
        "status": "ok" if svc.integrity.verified else "degraded",
        "brand": BRAND_NAME,
        "subtitle": BRAND_SUBTITLE,
        "ready": svc.ready and svc.integrity.verified,
        "integrity_verified": svc.integrity.verified,
        "integrity_status": svc.integrity.status,
        "device": str(svc.device),
        "local_only": True,
        "external_api": False,
        "persistence": "none",
    }


@app.get("/api/model")
def model() -> dict[str, Any]:
    return service().model_info()


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    svc = service()
    if not svc.integrity.verified:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_INTEGRITY_FAILED",
                "message": (
                    "Frozen model SHA256 verification failed. "
                    "Inference is refused to protect scientific integrity."
                ),
                "integrity_status": svc.integrity.status,
            },
        )

    filename = file.filename or "upload"
    lower = filename.lower()
    if not any(lower.endswith(ext) for ext in ALLOWED_EXT):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "UNSUPPORTED_FILE",
                "message": "Unsupported file type. Use JPEG, PNG, or WEBP.",
            },
        )

    content_type = (file.content_type or "").lower()
    # Some browsers omit/mislabel; still validate by content via PIL later
    if content_type and content_type not in ALLOWED_MIME and content_type != "application/octet-stream":
        raise HTTPException(
            status_code=400,
            detail={
                "code": "UNSUPPORTED_MIME",
                "message": "Unsupported MIME type. Use JPEG, PNG, or WEBP.",
            },
        )

    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=400,
            detail={"code": "EMPTY_FILE", "message": "Empty upload."},
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
            },
        )

    try:
        image = open_image_bytes(data)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "CORRUPT_IMAGE",
                "message": "Could not read image. File may be corrupt or unsupported.",
            },
        ) from None

    meta = {
        "filename": filename,
        "content_type": content_type or "unknown",
        "size_bytes": len(data),
        "width": int(image.width),
        "height": int(image.height),
        "mode": image.mode,
    }

    try:
        try:
            result = svc.analyze_pil(image, meta)
        except PermissionError as e:
            raise HTTPException(
                status_code=503,
                detail={"code": "MODEL_INTEGRITY_FAILED", "message": str(e)},
            ) from None
        except MemoryError:
            raise HTTPException(
                status_code=507,
                detail={
                    "code": "OUT_OF_MEMORY",
                    "message": "Not enough memory to run the frozen ensemble on this device.",
                },
            ) from None
        except Exception:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "INFERENCE_FAILURE",
                    "message": "Local inference failed. Please try another image or restart the demo.",
                },
            ) from None
        return JSONResponse(result)
    finally:
        # Do not retain uploaded image beyond the request
        try:
            image.close()
        except Exception:
            pass
        del data
