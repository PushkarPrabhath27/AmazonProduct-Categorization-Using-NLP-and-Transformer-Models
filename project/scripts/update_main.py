from pathlib import Path
path = Path('src/data_loader.py')
text = path.read_text()
text = text.replace('    df = load_products(args.products_path)\n', '    df = load_products(args.products_path, max_rows=args.max_rows)\n', 1)
path.write_text(text)
