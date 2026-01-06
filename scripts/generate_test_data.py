"""Replace the website placeholders with website domains from env_config
Generate the test data"""
import json

# Modified from the original version.
import os
import sys

# Add parent directory to path if running from outside webarena directory
script_dir = os.path.dirname(os.path.abspath(__file__))
webarena_dir = os.path.dirname(script_dir)
if webarena_dir not in sys.path:
    sys.path.insert(0, webarena_dir)


from browser_env.env_config import *


def main() -> None:
    # Modify from the original version.
    # Use paths relative to webarena directory
    config_dir = os.path.join(webarena_dir, "config_files")
    
    with open(os.path.join(config_dir, "test.raw.json"), "r", encoding="utf-8") as f:
        # test.raw.json is modified from the original version.
        raw = f.read()
    raw = raw.replace("__GITLAB__", GITLAB)
    raw = raw.replace("__REDDIT__", REDDIT)
    raw = raw.replace("__SHOPPING__", SHOPPING)
    raw = raw.replace("__SHOPPING_ADMIN__", SHOPPING_ADMIN)
    raw = raw.replace("__WIKIPEDIA__", WIKIPEDIA)
    raw = raw.replace("__MAP__", MAP)
    # Modified from the original version.
    with open(os.path.join(config_dir, "test.json"), "w", encoding="utf-8") as f:
        f.write(raw)
    # split to multiple files
    data = json.loads(raw)
    for idx, item in enumerate(data):
        # Modified from the original version.
        with open(os.path.join(config_dir, f"{idx}.json"), "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
