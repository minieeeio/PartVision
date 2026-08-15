import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from yolo_websocket import router as yolo_websocket_router
from yolo_config import yolo_settings

app = FastAPI(
    title="PartVision YOLO Backend",
    description="YOLOv8-seg real-time car part segmentation server",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(yolo_websocket_router)


@app.on_event("startup")
async def startup_event():
    print(f"[YOLO Main] Server starting on {yolo_settings.HOST}:{yolo_settings.PORT}")
    print(f"[YOLO Main] Model path: {yolo_settings.MODEL_PATH}")
    print(f"[YOLO Main] Device: {yolo_settings.DEVICE}")


if __name__ == "__main__":
    print("[YOLO Main] Starting uvicorn directly...")
    uvicorn.run(
        "yolo_main:app",
        host=yolo_settings.HOST,
        port=yolo_settings.PORT,
        reload=False,
    )
