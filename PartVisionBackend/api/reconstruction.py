import queue
import threading
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect

from core.reconstructor import get_manager, SessionStatus

router = APIRouter()
_manager = get_manager()


@router.post("/start")
def start_session():
    session = _manager.create_session()
    _manager.start_recording(session.session_id)
    return {
        "session_id": session.session_id,
        "status": session.status.value,
        "output_dir": session.output_dir,
    }


@router.post("/{session_id}/frame")
async def upload_frame(session_id: str, file: UploadFile = File(...)):
    session = _manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != SessionStatus.RECORDING:
        raise HTTPException(status_code=400, detail="Session is not recording")

    contents = await file.read()
    filename = _manager.add_frame(session_id, contents)
    if filename is None:
        raise HTTPException(status_code=400, detail="Failed to decode frame")

    return {
        "session_id": session_id,
        "frame": filename,
        "frame_count": session.frame_count,
    }


@router.post("/{session_id}/stop")
def stop_session(session_id: str):
    session = _manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != SessionStatus.RECORDING:
        raise HTTPException(status_code=400, detail="Session is not recording")

    _manager.stop_recording(session_id)
    thread = threading.Thread(target=_manager.run_pipeline, args=(session_id,), daemon=True)
    thread.start()

    return {
        "session_id": session_id,
        "status": SessionStatus.PROCESSING.value,
        "frame_count": session.frame_count,
    }


@router.post("/{session_id}/location")
def update_session_location(session_id: str, location: dict):
    session = _manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.gps = location
    return {"status": "ok"}


@router.get("/{session_id}/status")
def session_status(session_id: str):
    session = _manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "status": session.status.value,
        "frame_count": session.frame_count,
        "created_at": session.created_at,
        "completed_at": session.completed_at,
        "error_message": session.error_message,
        "output_dir": session.output_dir,
    }


@router.get("/{session_id}/files")
def list_files(session_id: str):
    session = _manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.output_dir:
        return {"files": []}

    base = Path(session.output_dir)
    files = []
    for path in base.rglob("*"):
        if path.is_file():
            files.append(str(path.relative_to(base)))
    return {"files": files}


@router.get("/{session_id}/download/{file_path:path}")
def download_file(session_id: str, file_path: str):
    session = _manager.get_session(session_id)
    if session is None or not session.output_dir:
        raise HTTPException(status_code=404, detail="Session not found")

    full_path = Path(session.output_dir) / file_path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return __import__("fastapi").responses.FileResponse(
        str(full_path), filename=full_path.name
    )


@router.websocket("/ws/reconstruction/{session_id}")
async def websocket_reconstruction_progress(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = _manager.get_session(session_id)
    if session is None:
        await websocket.close(code=4000)
        return

    q: queue.Queue = queue.Queue(maxsize=200)
    _manager.register_progress_queue(session_id, q)

    try:
        while True:
            try:
                msg = q.get(timeout=0.5)
                await websocket.send_json(msg)
                stage = msg.get("stage")
                if stage in ("done", "failed"):
                    break
            except queue.Empty:
                current = _manager.get_session(session_id)
                if current and current.status in (
                    SessionStatus.COMPLETED,
                    SessionStatus.FAILED,
                ):
                    break
                await websocket.send_json(
                    {"stage": "heartbeat", "status": session.status.value}
                )
    except WebSocketDisconnect:
        pass
    finally:
        _manager.unregister_progress_queue(session_id)
