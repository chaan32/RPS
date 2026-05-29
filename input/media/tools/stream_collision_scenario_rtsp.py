"""Publish recorded Unity collision scenarios as RTSP camera streams.

This is the bridge between the Unity-generated scenario frames and the
backend realtime loop. It publishes:

  cam1_frames -> rtsp://localhost:8554/cam1
  cam2_frames -> rtsp://localhost:8554/cam2

MediaMTX must be running before this script starts.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = PROJECT_ROOT / "simulation" / "Recordings" / "collision_scenarios"
DEFAULT_SCENARIO = "scenario_01_center_crossing"


@dataclass(frozen=True)
class FrameSet:
    cam_id: str
    directory: Path
    count: int
    first: int
    last: int


def parse_recording_info(path: Path) -> dict[str, str]:
    info_path = path / "recording_info.txt"
    if not info_path.exists():
        return {}

    info: dict[str, str] = {}
    for raw in info_path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        info[key.strip()] = value.strip()
    return info


def resolve_scenario(root: Path, scenario: str) -> Path:
    candidate = Path(scenario).expanduser()
    if candidate.exists():
        return candidate.resolve()

    candidate = root / scenario
    if candidate.exists():
        return candidate.resolve()

    available = ", ".join(sorted(p.name for p in root.iterdir() if p.is_dir()))
    raise FileNotFoundError(
        f"Scenario not found: {scenario}\n"
        f"Root: {root}\n"
        f"Available: {available or '(none)'}"
    )


def read_fps(info: dict[str, str], fallback: float) -> float:
    raw = info.get("fps")
    if not raw:
        return fallback
    try:
        fps = float(raw)
    except ValueError:
        return fallback
    return fps if fps > 0 else fallback


def resolve_frame_dir(scenario_path: Path, info: dict[str, str], cam_id: str) -> Path:
    fallback = scenario_path / f"{cam_id}_frames"
    if fallback.exists():
        return fallback.resolve()

    key = f"{cam_id}_frames"
    if key in info:
        configured = Path(info[key]).expanduser()
        if configured.exists():
            return configured.resolve()

    raise FileNotFoundError(f"{cam_id} frame directory not found under {scenario_path}")


def inspect_frames(cam_id: str, directory: Path) -> FrameSet:
    pattern = re.compile(r"frame_(\d{4})\.jpg$")
    frame_numbers = []
    for frame in directory.glob("*.jpg"):
        match = pattern.fullmatch(frame.name)
        if match:
            frame_numbers.append(int(match.group(1)))

    if not frame_numbers:
        raise FileNotFoundError(f"No canonical frame_0000.jpg frames in {directory}")

    frame_numbers.sort()
    expected = list(range(frame_numbers[0], frame_numbers[-1] + 1))
    if frame_numbers != expected:
        missing = sorted(set(expected) - set(frame_numbers))
        preview = ", ".join(f"frame_{n:04d}.jpg" for n in missing[:8])
        raise RuntimeError(
            f"{cam_id} frames are not contiguous in {directory}. "
            f"Missing: {preview or '(unknown)'}"
        )
    if frame_numbers[0] != 0:
        raise RuntimeError(
            f"{cam_id} frames must start at frame_0000.jpg for FFmpeg sequence input: "
            f"first=frame_{frame_numbers[0]:04d}.jpg"
        )

    return FrameSet(
        cam_id=cam_id,
        directory=directory,
        count=len(frame_numbers),
        first=frame_numbers[0],
        last=frame_numbers[-1],
    )


def build_ffmpeg_command(
    ffmpeg_bin: str,
    frame_set: FrameSet,
    fps: float,
    rtsp_url: str,
    loglevel: str,
) -> list[str]:
    gop = max(1, int(round(fps * 2)))
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        loglevel,
        "-nostdin",
        "-re",
        "-stream_loop",
        "-1",
        "-framerate",
        f"{fps:g}",
        "-start_number",
        "0",
        "-i",
        str(frame_set.directory / "frame_%04d.jpg"),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(gop),
        "-bf",
        "0",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        rtsp_url,
    ]


def terminate_processes(processes: list[subprocess.Popen]) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if all(proc.poll() is not None for proc in processes):
            return
        time.sleep(0.1)

    for proc in processes:
        if proc.poll() is None:
            proc.kill()


def run_bridge(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    scenario_path = resolve_scenario(root, args.scenario)
    info = parse_recording_info(scenario_path)
    fps = read_fps(info, args.fps)

    cam1 = inspect_frames("cam1", resolve_frame_dir(scenario_path, info, "cam1"))
    cam2 = inspect_frames("cam2", resolve_frame_dir(scenario_path, info, "cam2"))

    ffmpeg_bin = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise FileNotFoundError("ffmpeg not found. Install ffmpeg or pass --ffmpeg.")

    rtsp_base = args.rtsp_base.rstrip("/")
    targets = {
        "cam1": f"{rtsp_base}/{args.cam1_path.lstrip('/')}",
        "cam2": f"{rtsp_base}/{args.cam2_path.lstrip('/')}",
    }

    print(f"[rtsp-bridge] scenario: {scenario_path.name}")
    print(f"[rtsp-bridge] fps: {fps:g}")
    print(
        f"[rtsp-bridge] cam1 frames: {cam1.count} "
        f"({cam1.directory / 'frame_0000.jpg'} -> frame_{cam1.last:04d}.jpg)"
    )
    print(
        f"[rtsp-bridge] cam2 frames: {cam2.count} "
        f"({cam2.directory / 'frame_0000.jpg'} -> frame_{cam2.last:04d}.jpg)"
    )
    print(f"[rtsp-bridge] publish cam1: {targets['cam1']}")
    print(f"[rtsp-bridge] publish cam2: {targets['cam2']}")

    commands = [
        build_ffmpeg_command(ffmpeg_bin, cam1, fps, targets["cam1"], args.loglevel),
        build_ffmpeg_command(ffmpeg_bin, cam2, fps, targets["cam2"], args.loglevel),
    ]
    if args.print_commands:
        for command in commands:
            print("[rtsp-bridge] command:", shlex.join(command))

    if args.validate_only:
        print("[rtsp-bridge] validate-only complete")
        return 0

    processes = [subprocess.Popen(command) for command in commands]
    stop = False

    def handle_signal(signum, frame):  # noqa: ARG001
        nonlocal stop
        stop = True

    previous_sigint = signal.signal(signal.SIGINT, handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_signal)

    started_at = time.time()
    try:
        while not stop:
            for proc in processes:
                code = proc.poll()
                if code is not None:
                    print(f"[rtsp-bridge] ffmpeg exited with code {code}")
                    return code

            if args.duration > 0 and time.time() - started_at >= args.duration:
                failed = [
                    proc.poll()
                    for proc in processes
                    if proc.poll() not in (None, 0)
                ]
                if failed:
                    print(f"[rtsp-bridge] ffmpeg failed before duration ended: {failed}")
                    return int(failed[0])
                print(f"[rtsp-bridge] duration reached: {args.duration:g}s")
                return 0

            time.sleep(0.2)
    finally:
        terminate_processes(processes)
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        help="Scenario folder name under collision_scenarios, or an absolute path.",
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--rtsp-base", default=os.getenv("RTSP_PUBLISH_BASE", "rtsp://localhost:8554"))
    parser.add_argument("--cam1-path", default="cam1")
    parser.add_argument("--cam2-path", default="cam2")
    parser.add_argument("--ffmpeg", default=os.getenv("FFMPEG_BIN", ""))
    parser.add_argument(
        "--loglevel",
        default="warning",
        choices=["quiet", "panic", "fatal", "error", "warning", "info", "verbose", "debug"],
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after N seconds. 0 means run until Ctrl-C.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate scenario frames and print target URLs without starting FFmpeg.",
    )
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()

    try:
        return run_bridge(args)
    except Exception as exc:
        print(f"[rtsp-bridge] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
