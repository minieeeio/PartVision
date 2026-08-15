import os
import queue
import shutil
import time
import uuid
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


class SessionStatus(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReconstructionSession:
    session_id: str
    status: SessionStatus = SessionStatus.IDLE
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    frame_count: int = 0
    output_dir: Optional[str] = None
    error_message: Optional[str] = None
    completed_at: Optional[str] = None
    matching_type: str = "sequential"
    dense: bool = False
    gps: Optional[Dict[str, List[float]]] = None
    rejected_frames: int = 0


class ReconstructionManager:
    def __init__(self, base_dir: str = "reconstructions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, ReconstructionSession] = {}
        self._lock = threading.Lock()
        self._progress_queues: Dict[str, "queue.Queue"] = {}
        self._cuda_available: Optional[bool] = None

    def _check_cuda(self) -> bool:
        if self._cuda_available is None:
            try:
                import pycolmap
                self._cuda_available = bool(pycolmap.has_cuda)
            except Exception:
                self._cuda_available = False
        return self._cuda_available

    def create_session(
        self,
        matching_type: str = "sequential",
        dense: bool = False,
        gps: Optional[Dict[str, List[float]]] = None,
    ) -> ReconstructionSession:
        session_id = uuid.uuid4().hex
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "images").mkdir(parents=True, exist_ok=True)

        session = ReconstructionSession(
            session_id=session_id,
            status=SessionStatus.IDLE,
            output_dir=str(session_dir),
            matching_type=matching_type,
            dense=dense,
            gps=gps or {},
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ReconstructionSession]:
        return self._sessions.get(session_id)

    def _compute_blur_score(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def add_frame(self, session_id: str, frame_bytes: bytes) -> Optional[str]:
        from config import settings

        session = self._sessions.get(session_id)
        if session is None or session.status != SessionStatus.RECORDING:
            return None

        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        if self._compute_blur_score(frame) < settings.BLUR_THRESHOLD:
            session.rejected_frames += 1
            return None

        image_dir = Path(session.output_dir) / "images"
        filename = f"{session.frame_count:06d}.jpg"
        filepath = image_dir / filename
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        session.frame_count += 1
        return filename

    def start_recording(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None or session.status != SessionStatus.IDLE:
            return False
        session.status = SessionStatus.RECORDING
        return True

    def stop_recording(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None or session.status != SessionStatus.RECORDING:
            return False
        session.status = SessionStatus.PROCESSING
        return True

    def mark_completed(self, session_id: str):
        session = self._sessions.get(session_id)
        if session:
            session.status = SessionStatus.COMPLETED
            session.completed_at = datetime.now().isoformat()

    def mark_failed(self, session_id: str, error: str):
        session = self._sessions.get(session_id)
        if session:
            session.status = SessionStatus.FAILED
            session.error_message = error
            session.completed_at = datetime.now().isoformat()

    def register_progress_queue(self, session_id: str, q: "queue.Queue"):
        self._progress_queues[session_id] = q

    def unregister_progress_queue(self, session_id: str):
        self._progress_queues.pop(session_id, None)

    def _progress(self, session_id: str, message: Dict[str, Any]):
        q = self._progress_queues.get(session_id)
        if q is not None:
            try:
                q.put_nowait(message)
            except queue.Full:
                pass

    def _run_dense_pipeline(
        self,
        colmap_bin: str,
        image_dir: str,
        database_path: str,
        sparse_path: str,
        dense_path: str,
        mesh_path: str,
        session_id: str,
    ):
        import subprocess

        workspace = Path(dense_path)
        workspace.mkdir(parents=True, exist_ok=True)

        self._progress(session_id, {"stage": "dense", "progress": 10, "detail": "undistort"})
        subprocess.run(
            [
                colmap_bin,
                "image_undistorter",
                "--image_path", image_dir,
                "--input_path", sparse_path,
                "--output_path", str(workspace),
                "--output_type", "COLMAP",
            ],
            check=True,
            capture_output=True,
        )

        self._progress(session_id, {"stage": "dense", "progress": 30, "detail": "patch_match_stereo"})
        subprocess.run(
            [
                colmap_bin,
                "patch_match_stereo",
                "--workspace_path", str(workspace),
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.geom_consistency", "true",
            ],
            check=True,
            capture_output=True,
        )

        self._progress(session_id, {"stage": "dense", "progress": 60, "detail": "stereo_fusion"})
        import pycolmap

        fusion_opts = pycolmap.StereoFusionOptions()
        pycolmap.stereo_fusion(
            output_path=str(workspace / "fused.ply"),
            workspace_path=str(workspace),
            options=fusion_opts,
        )

        self._progress(session_id, {"stage": "dense", "progress": 80, "detail": "poisson_meshing"})
        Path(mesh_path).mkdir(parents=True, exist_ok=True)
        poisson_opts = pycolmap.PoissonMeshingOptions()
        pycolmap.poisson_meshing(
            input_path=str(workspace / "fused.ply"),
            output_path=str(Path(mesh_path) / "mesh.ply"),
            options=poisson_opts,
        )
        self._progress(session_id, {"stage": "dense", "progress": 100})

    def run_pipeline(self, session_id: str):
        session = self._sessions.get(session_id)
        if not session or session.status != SessionStatus.PROCESSING:
            return

        try:
            image_dir = str(Path(session.output_dir) / "images")
            database_path = str(Path(session.output_dir) / "database.db")
            sparse_path = str(Path(session.output_dir) / "sparse")
            dense_path = str(Path(session.output_dir) / "dense")
            mesh_path = str(Path(session.output_dir) / "mesh")
            Path(sparse_path).mkdir(parents=True, exist_ok=True)

            import pycolmap

            device = pycolmap.Device.cuda if self._check_cuda() else pycolmap.Device.cpu

            self._progress(session_id, {"stage": "extracting_features", "progress": 0})
            sift_opts = pycolmap.SiftExtractionOptions()
            if self._check_cuda():
                sift_opts.gpu_index = 0
            pycolmap.extract_features(
                database_path, image_dir, sift_options=sift_opts, device=device
            )
            self._progress(session_id, {"stage": "extracting_features", "progress": 100})

            self._progress(session_id, {"stage": "matching", "progress": 0})
            if session.matching_type == "sequential":
                seq_opts = pycolmap.SequentialMatchingOptions()
                seq_opts.overlap = 10
                match_opts = pycolmap.SiftMatchingOptions()
                if self._check_cuda():
                    match_opts.gpu_index = 0
                pycolmap.match_sequential(
                    database_path,
                    sift_options=match_opts,
                    matching_options=seq_opts,
                    device=device,
                )
            else:
                pycolmap.match_exhaustive(database_path)
            self._progress(session_id, {"stage": "matching", "progress": 100})

            self._progress(session_id, {"stage": "mapping", "progress": 0})
            mapper_opts = pycolmap.IncrementalMapperOptions()
            maps = pycolmap.incremental_mapping(
                database_path, image_dir, sparse_path, mapper_opts
            )
            if not maps:
                self.mark_failed(session_id, "No reconstruction produced")
                self._progress(
                    session_id,
                    {"stage": "failed", "progress": 100, "error": "No reconstruction produced"},
                )
                return

            best_map = max(maps.values(), key=lambda m: len(m.reg_image_ids()))
            best_map.write(sparse_path)
            best_map.export_PLY(str(Path(sparse_path) / "sparse.ply"))
            self._progress(session_id, {"stage": "mapping", "progress": 100})

            if session.gps:
                try:
                    import numpy as np

                    image_names = [
                        img.name
                        for img in best_map.images.values()
                        if img.name in session.gps
                    ]
                    locations = [
                        np.array([[v[0]], [v[1]], [v[2]]], dtype=np.float64)
                        for v in [session.gps[name] for name in image_names]
                    ]
                    if image_names:
                        sim3 = pycolmap.align_reconstrution_to_locations(
                            best_map,
                            image_names,
                            locations,
                            min_common_points=3,
                            ransac_options=pycolmap.RANSACOptions(),
                        )
                        best_map.transform(sim3)
                        best_map.write(sparse_path)
                        best_map.export_PLY(str(Path(sparse_path) / "sparse.ply"))
                except Exception as e:
                    print(f"[Reconstruction] GPS alignment failed: {e}")

            if session.dense and shutil.which("colmap"):
                self._run_dense_pipeline(
                    shutil.which("colmap"),
                    image_dir,
                    database_path,
                    sparse_path,
                    dense_path,
                    mesh_path,
                    session_id,
                )

            self.mark_completed(session_id)
            self._progress(session_id, {"stage": "done", "progress": 100})

        except Exception:
            self.mark_failed(session_id, traceback.format_exc())
            self._progress(
                session_id,
                {"stage": "failed", "progress": 100, "error": traceback.format_exc()},
            )


_manager = ReconstructionManager()


def get_manager() -> ReconstructionManager:
    return _manager
