import asyncio
import torch
import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.websocket import router as websocket_router
from app.api.websocket import model_wrapper
from app.core.logging_config import setup_logging
from app.core.connection_manager import connection_manager
from config import settings

setup_logging("INFO")
logger = logging.getLogger("PartVision")

# 1. Initialize FastAPI Application Instance
app = FastAPI(
    title="PartVision AI Backend",
    description="Real-time car part segmentation and detection server for CoreScan",
    version="1.0.0"
)

# 2. Configure Cross-Origin Resource Sharing (CORS) Rules
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Mount WebSocket API Router
app.include_router(websocket_router)


# 4. Warm up the model on startup (initializes CUDA kernels, avoids 1st-frame latency)
@app.on_event("startup")
async def warmup_model():
    try:
        dummy_input = torch.randn(1, 3, settings.INPUT_SIZE[1], settings.INPUT_SIZE[0])
        await asyncio.to_thread(model_wrapper.predict, dummy_input)
        logger.info("Model warmup completed — device: %s", settings.DEVICE)
    except Exception as e:
        logger.error("Model warmup failed: %s", e)


# 5. Health-Check Endpoint for Server Diagnostics
@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "device": settings.DEVICE,
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
        "active_connections": connection_manager.active_count,
        "input_size": list(settings.INPUT_SIZE),
    }


# 6. Standalone Execution Entry Point
if __name__ == "__main__":
    logger.info("Starting PartVision backend on %s:%s", settings.HOST, settings.PORT)
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
