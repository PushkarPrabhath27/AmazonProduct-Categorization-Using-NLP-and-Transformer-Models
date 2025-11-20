from pathlib import Path
path = Path('src/data_loader.py')
text = path.read_text()
text = text.replace('    logging.basicConfig(\n        level=logging.INFO,\n        format="%(asctime)s - %(levelname)s - %(message)s",\n        handlers=[', '    logging.basicConfig(\n        level=logging.INFO,\n        format="%(asctime)s - %(levelname)s - %(message)s",\n        handlers=[', 1)
if 'force=True' not in text:
    text = text.replace('        handlers=[', '        handlers[', 1)
path.write_text(text)
