from pathlib import Path
path = Path('src/data_loader.py')
text = path.read_text()
text = text.replace('["category", "category_name", "label"]', '["category", "category_id", "category_name", "label"]', 1)
path.write_text(text)
