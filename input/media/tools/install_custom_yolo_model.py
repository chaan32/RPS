"""Install a newly trained custom YOLO model into model/yolo.

Accepts either a direct best.pt file or a Colab result zip that contains
`weights/best.pt`. The previous target model is backed up before replacement.

Example:
    python input/media/tools/install_custom_yolo_model.py \
      --source ~/Downloads/forklift_box_unity_color_result.zip
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGET = PROJECT_ROOT / "model" / "yolo" / "best_forklift_box_colab.pt"


def find_best_pt_in_zip(source: Path) -> tuple[tempfile.TemporaryDirectory, Path]:
    tempdir = tempfile.TemporaryDirectory()
    temp_path = Path(tempdir.name)
    with zipfile.ZipFile(source, "r") as zf:
        members = [name for name in zf.namelist() if name.endswith("/weights/best.pt")]
        if not members:
            members = [name for name in zf.namelist() if name.endswith("best.pt")]
        if not members:
            tempdir.cleanup()
            raise FileNotFoundError(f"best.pt not found inside zip: {source}")
        member = sorted(members, key=len)[0]
        zf.extract(member, temp_path)
    return tempdir, temp_path / member


def resolve_source(source: Path) -> tuple[tempfile.TemporaryDirectory | None, Path]:
    if source.suffix.lower() == ".zip":
        return find_best_pt_in_zip(source)
    if source.name != "best.pt" and source.suffix.lower() != ".pt":
        raise ValueError(f"source must be a .pt file or result .zip: {source}")
    if not source.exists():
        raise FileNotFoundError(source)
    return None, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="best.pt or Colab result zip")
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    target = Path(args.target).expanduser().resolve()
    tempdir, best_pt = resolve_source(source)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = target.with_name(f"{target.stem}.backup_{stamp}{target.suffix}")
            shutil.copy2(target, backup)
            print(f"backup={backup}")

        shutil.copy2(best_pt, target)
        print(f"installed={target}")
        print(f"source_best={best_pt}")
    finally:
        if tempdir is not None:
            tempdir.cleanup()


if __name__ == "__main__":
    main()
