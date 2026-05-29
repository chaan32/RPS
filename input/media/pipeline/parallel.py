"""Parallel detection executors for realtime benchmarking.

The realtime loop owns camera IO, fusion, DB, and visualization. This module
only owns the expensive image-to-detections step so serial, 2-thread, and
4-thread modes can be compared with the same outer pipeline.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from .engine import DetectionPipeline


PipelineFactory = Callable[..., DetectionPipeline]


@dataclass
class PairExtraction:
    cam1: list[dict]
    cam2: list[dict]
    cam1_timing: dict[str, float | int | str]
    cam2_timing: dict[str, float | int | str]


def _empty_timing(cam_id: str) -> dict[str, float | int | str]:
    return {
        "cam_id": cam_id,
        "extract_total_ms": 0.0,
        "pose_track_ms": 0.0,
        "custom_yolo_ms": 0.0,
        "aruco_detect_ms": 0.0,
        "worker_collect_ms": 0.0,
        "custom_postprocess_ms": 0.0,
        "detections_total": 0,
    }


def _merge_timing(
    cam_id: str,
    worker_timing: dict[str, float | int | str],
    custom_timing: dict[str, float | int | str],
    *,
    worker_wall_ms: float,
    custom_wall_ms: float,
    detections_total: int,
) -> dict[str, float | int | str]:
    timing = {"cam_id": cam_id}
    for source in (worker_timing, custom_timing):
        for key, value in source.items():
            if key != "cam_id":
                timing[key] = value
    timing["split_worker_wall_ms"] = worker_wall_ms
    timing["split_custom_wall_ms"] = custom_wall_ms
    timing["extract_total_ms"] = max(worker_wall_ms, custom_wall_ms)
    timing["detections_total"] = detections_total
    return timing


class DetectionExecutor:
    """Common API used by realtime_camera for serial/parallel extraction."""

    def __init__(self, mode: str, pipeline_factory: PipelineFactory):
        self.mode = mode
        self._executor: ThreadPoolExecutor | None = None
        self._coordinator: DetectionPipeline

        if mode == "serial":
            self._coordinator = pipeline_factory()
            self._pipelines = {"cam1": self._coordinator, "cam2": self._coordinator}
            self._pose_pipelines = {}
            self._custom_pipelines = {}
        elif mode == "camera_parallel":
            shared_track_state: dict[str, dict[int, str]] = {"cam1": {}, "cam2": {}}
            shared_world_state: dict[str, dict] = {}
            self._pipelines = {
                "cam1": pipeline_factory(),
                "cam2": pipeline_factory(),
            }
            for pipeline in self._pipelines.values():
                pipeline.cam_track_to_worker = shared_track_state
                pipeline.worker_world_state = shared_world_state
            self._coordinator = self._pipelines["cam1"]
            self._pose_pipelines = {}
            self._custom_pipelines = {}
            self._executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="camera-extract",
            )
        elif mode == "model_parallel":
            shared_track_state = {"cam1": {}, "cam2": {}}
            shared_world_state = {}
            self._pose_pipelines = {
                "cam1": pipeline_factory(load_custom=False),
                "cam2": pipeline_factory(load_custom=False),
            }
            for pipeline in self._pose_pipelines.values():
                pipeline.cam_track_to_worker = shared_track_state
                pipeline.worker_world_state = shared_world_state
            self._custom_pipelines = {
                "cam1": pipeline_factory(load_pose=False),
                "cam2": pipeline_factory(load_pose=False),
            }
            self._pipelines = {}
            self._coordinator = self._pose_pipelines["cam1"]
            self._executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="model-extract",
            )
        else:
            raise ValueError(
                "mode must be one of: serial, camera_parallel, model_parallel"
            )

    def extract_pair(
        self,
        cam1_frame,
        cam1_ok: bool,
        cam2_frame,
        cam2_ok: bool,
    ) -> PairExtraction:
        if self.mode == "serial":
            return self._extract_serial(cam1_frame, cam1_ok, cam2_frame, cam2_ok)
        if self.mode == "camera_parallel":
            return self._extract_camera_parallel(
                cam1_frame, cam1_ok, cam2_frame, cam2_ok,
            )
        return self._extract_model_parallel(cam1_frame, cam1_ok, cam2_frame, cam2_ok)

    def cross_camera_propagate(
        self,
        detections_by_cam: dict[str, list[dict]],
        now_ts: float,
    ) -> None:
        self._coordinator.cross_camera_propagate(detections_by_cam, now_ts=now_ts)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def _extract_serial(
        self,
        cam1_frame,
        cam1_ok: bool,
        cam2_frame,
        cam2_ok: bool,
    ) -> PairExtraction:
        d1: list[dict] = []
        d2: list[dict] = []
        cam1_timing = _empty_timing("cam1")
        cam2_timing = _empty_timing("cam2")
        if cam1_ok and cam1_frame is not None:
            d1 = self._coordinator.extract(cam1_frame, "cam1")
            cam1_timing = self._coordinator.get_last_timing("cam1")
        if cam2_ok and cam2_frame is not None:
            d2 = self._coordinator.extract(cam2_frame, "cam2")
            cam2_timing = self._coordinator.get_last_timing("cam2")
        return PairExtraction(d1, d2, cam1_timing, cam2_timing)

    def _extract_camera_parallel(
        self,
        cam1_frame,
        cam1_ok: bool,
        cam2_frame,
        cam2_ok: bool,
    ) -> PairExtraction:
        assert self._executor is not None

        futures: dict[str, Future[tuple[list[dict], dict[str, float | int | str]]]] = {}
        if cam1_ok and cam1_frame is not None:
            futures["cam1"] = self._executor.submit(
                self._extract_full_camera, "cam1", cam1_frame,
            )
        if cam2_ok and cam2_frame is not None:
            futures["cam2"] = self._executor.submit(
                self._extract_full_camera, "cam2", cam2_frame,
            )

        d1, cam1_timing = futures["cam1"].result() if "cam1" in futures else ([], _empty_timing("cam1"))
        d2, cam2_timing = futures["cam2"].result() if "cam2" in futures else ([], _empty_timing("cam2"))
        return PairExtraction(d1, d2, cam1_timing, cam2_timing)

    def _extract_full_camera(
        self,
        cam_id: str,
        frame,
    ) -> tuple[list[dict], dict[str, float | int | str]]:
        pipeline = self._pipelines[cam_id]
        detections = pipeline.extract(frame, cam_id)
        return detections, pipeline.get_last_timing(cam_id)

    def _extract_model_parallel(
        self,
        cam1_frame,
        cam1_ok: bool,
        cam2_frame,
        cam2_ok: bool,
    ) -> PairExtraction:
        assert self._executor is not None

        futures: dict[tuple[str, str], Future] = {}
        if cam1_ok and cam1_frame is not None:
            futures[("cam1", "worker")] = self._executor.submit(
                self._extract_workers_task, "cam1", cam1_frame,
            )
            futures[("cam1", "custom")] = self._executor.submit(
                self._extract_custom_task, "cam1", cam1_frame,
            )
        if cam2_ok and cam2_frame is not None:
            futures[("cam2", "worker")] = self._executor.submit(
                self._extract_workers_task, "cam2", cam2_frame,
            )
            futures[("cam2", "custom")] = self._executor.submit(
                self._extract_custom_task, "cam2", cam2_frame,
            )

        d1, cam1_timing = self._collect_split_result("cam1", futures)
        d2, cam2_timing = self._collect_split_result("cam2", futures)
        return PairExtraction(d1, d2, cam1_timing, cam2_timing)

    def _collect_split_result(
        self,
        cam_id: str,
        futures: dict[tuple[str, str], Future],
    ) -> tuple[list[dict], dict[str, float | int | str]]:
        worker_future = futures.get((cam_id, "worker"))
        custom_future = futures.get((cam_id, "custom"))
        if worker_future is None and custom_future is None:
            return [], _empty_timing(cam_id)

        worker_entries, worker_timing, worker_wall_ms = (
            worker_future.result() if worker_future is not None else ([], {}, 0.0)
        )
        custom_entries, custom_timing, custom_wall_ms = (
            custom_future.result() if custom_future is not None else ([], {}, 0.0)
        )
        detections = worker_entries + custom_entries
        timing = _merge_timing(
            cam_id,
            worker_timing,
            custom_timing,
            worker_wall_ms=worker_wall_ms,
            custom_wall_ms=custom_wall_ms,
            detections_total=len(detections),
        )
        return detections, timing

    def _extract_workers_task(
        self,
        cam_id: str,
        frame,
    ) -> tuple[list[dict], dict[str, float | int | str], float]:
        started = time.perf_counter()
        entries, timing = self._pose_pipelines[cam_id].extract_workers(frame, cam_id)
        return entries, timing, (time.perf_counter() - started) * 1000.0

    def _extract_custom_task(
        self,
        cam_id: str,
        frame,
    ) -> tuple[list[dict], dict[str, float | int | str], float]:
        started = time.perf_counter()
        entries, timing = self._custom_pipelines[cam_id].extract_custom(frame, cam_id)
        return entries, timing, (time.perf_counter() - started) * 1000.0


def build_detection_executor(
    mode: str,
    pipeline_factory: PipelineFactory,
) -> DetectionExecutor:
    return DetectionExecutor(mode, pipeline_factory)
