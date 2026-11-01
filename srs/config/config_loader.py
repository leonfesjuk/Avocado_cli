import json
from pathlib import Path

CONFIG_FILE_PATH = Path(__file__).resolve().parent.parent/"config.json"

def load_config(path: Path = CONFIG_FILE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
    
CONFIG = load_config()

#Config variables
PROJECT_NAME = CONFIG["project_name"]