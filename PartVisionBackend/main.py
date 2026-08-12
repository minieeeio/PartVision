import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.websocket import router as websocket_router
from config import settings

# 1. Initialize FastAPI Application Instance
app = FastAPI(
    title="PartVision AI Backend",
    description="Real-time car part segmentation and detection server for CoreScan iOS",
    version="1.0.0"
)

# 2. Configure Cross-Origin Resource Sharing (CORS) Rules
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust allowed origins in strict production environments
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
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD
    }

# 5. Standalone Execution Entry Point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True  # Enables auto-reloading during backend development
    )