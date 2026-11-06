import os
from pathlib import Path

os.environ["AVOCADO_CONFIG_PATH"] = str(
    Path(__file__).resolve().parents[1] / "srs" / "config" / "config.json"
)
