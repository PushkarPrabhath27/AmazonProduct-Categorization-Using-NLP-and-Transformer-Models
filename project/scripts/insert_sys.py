from pathlib import Path

path = Path('src/data_loader.py')
text = path.read_text()
if 'import sys' not in text.splitlines()[12:20]:
    text = text.replace('import logging\nfrom pathlib import Path', 'import logging\nimport sys\nfrom pathlib import Path', 1)
path.write_text(text)
