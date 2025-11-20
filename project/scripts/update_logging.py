from pathlib import Path
path = Path('src/data_loader.py')
text = path.read_text()
text = text.replace('logging.StreamHandler()', 'logging.StreamHandler(stream=sys.stdout)', 1)
path.write_text(text)
