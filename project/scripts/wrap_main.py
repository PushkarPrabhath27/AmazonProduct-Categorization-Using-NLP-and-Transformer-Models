from pathlib import Path
path = Path('src/data_loader.py')
text = path.read_text()
old = "def main() -> None:\n    setup_logging()\n    args = parse_args()\n\n    df = load_products(args.products_path, max_rows=args.max_rows)\n    df = canonicalize_columns(df)\n    df = merge_categories(df, args.categories_path)\n    write_output(df, args.output_path)\n\n\nif __name__ == \"__main__\":\n    main()\n"
new = "def main() -> None:\n    setup_logging()\n    args = parse_args()\n\n    try:\n        df = load_products(args.products_path, max_rows=args.max_rows)\n        df = canonicalize_columns(df)\n        df = merge_categories(df, args.categories_path)\n        write_output(df, args.output_path)\n    except Exception as exc:\n        logging.exception(\"data_loader failed\")\n        raise\n\n\nif __name__ == \"__main__\":\n    main()\n"
if old not in text:
    raise SystemExit('main block not found')
path.write_text(text.replace(old, new))
