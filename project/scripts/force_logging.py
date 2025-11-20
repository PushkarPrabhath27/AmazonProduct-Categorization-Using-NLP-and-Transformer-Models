from pathlib import Path
path = Path('src/data_loader.py')
text = path.read_text()
old = "    logging.basicConfig(\n        level=logging.INFO,\n        format=\"%(asctime)s - %(levelname)s - %(message)s\",\n        handlers=[\n            logging.FileHandler(LOG_FILE, encoding=\"utf-8\"),\n            logging.StreamHandler(stream=sys.stdout),\n        ],\n    )\n"
new = "    logging.basicConfig(\n        level=logging.INFO,\n        format=\"%(asctime)s - %(levelname)s - %(message)s\",\n        handlers=[\n            logging.FileHandler(LOG_FILE, encoding=\"utf-8\"),\n            logging.StreamHandler(stream=sys.stdout),\n        ],\n        force=True,\n    )\n"
if old not in text:
    raise SystemExit('logging block not found')
path.write_text(text.replace(old, new))
