import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.websocket import router as websocket_router
from api.reconstruction import router as reconstruction_router
from core.metrics import start_monitoring, get_resource_monitor
from config import settings
from model_manager import model_manager

app = FastAPI(
    title="PartVision AI Backend",
    description="Real-time car part segmentation and detection server",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websocket_router)
app.include_router(reconstruction_router)


@app.post("/switch_model")
def switch_model(model_type: str):
    print(f"[Main] /switch_model called with model_type={model_type}")
    result = model_manager.switch_model(model_type)
    print(f"[Main] /switch_model result: {result}")
    return result


@app.get("/models")
def list_models():
    return {
        "current_model": model_manager.get_current_model_type(),
        "available_models": model_manager.get_available_models(),
    }


@app.on_event("startup")
async def startup_event():
    print(f"[Main] Server starting on {settings.HOST}:{settings.PORT}")
    print(f"[Main] Model path: {settings.MODEL_PATH}")
    print(f"[Main] Device: {settings.DEVICE}")
    print(f"[Main] Current model: {model_manager.get_current_model_type()}")
    start_monitoring()
    print("[Main] Resource monitoring started.")


@app.on_event("shutdown")
async def shutdown_event():
    get_resource_monitor().stop()
    print("[Main] Resource monitoring stopped.")


if __name__ == "__main__":
    print("[Main] Starting uvicorn directly...")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
    )
