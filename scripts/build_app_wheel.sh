#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/app"
VENDOR_DIR="$APP_DIR/vendor"
REQ_FILE="$APP_DIR/requirements.txt"

cleanup() {
  rm -rf "$ROOT_DIR/build" "$ROOT_DIR/dist" "$ROOT_DIR/memory_agent.egg-info"
}

trap cleanup EXIT

cd "$ROOT_DIR"

rm -rf "$VENDOR_DIR"
mkdir -p "$VENDOR_DIR"

python -m build --wheel --outdir "$VENDOR_DIR" "$ROOT_DIR"

WHEEL_FILENAME="$(python - <<'PY'
from pathlib import Path
import tomllib

root = Path.cwd()
pyproject = tomllib.loads((root / "pyproject.toml").read_text())
version = pyproject["project"]["version"]
wheels = sorted((root / "app" / "vendor").glob(f"memory_agent-{version}-*.whl"))

if len(wheels) != 1:
    raise SystemExit(
        f"Expected exactly one wheel for version {version}, found {len(wheels)}"
    )

print(wheels[0].name)
PY
)"

python - <<'PY' "$REQ_FILE" "$WHEEL_FILENAME"
from pathlib import Path
import sys

req_path = Path(sys.argv[1])
wheel_filename = sys.argv[2]
lines = req_path.read_text().splitlines()

if not lines:
    raise SystemExit(f"{req_path} is empty")

lines[-1] = f"./vendor/{wheel_filename}"
req_path.write_text("\n".join(lines) + "\n")
PY
