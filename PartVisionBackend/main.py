import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.websocket import router as websocket_router
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

# 4. Health-Check Endpoint for Server Diagnostics
@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "device": settings.DEVICE,
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
        "active_connections": connection_manager.active_count,
    }

# 5. Standalone Execution Entry Point
if __name__ == "__main__":
    logger.info("Starting PartVision backend on %s:%s", settings.HOST, settings.PORT)
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )