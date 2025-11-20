from pathlib import Path

path = Path('src/data_loader.py')
text = path.read_text()
needle = "parser.add_argument(\n        \"--output-path\",\n        type=Path,\n        default=RAW_DIR / \"amazon_products_clean.csv\",\n        help=\"Output CSV with canonical columns\",\n    )\n    return parser.parse_args()"
if needle not in text:
    raise SystemExit('parser block not found')
insert = "parser.add_argument(\n        \"--max-rows\",\n        type=int,\n        default=None,\n        help=\"Optional row limit for development / memory-constrained runs\",\n    )\n"
text = text.replace(needle, needle.replace("    return parser.parse_args()", """    parser.add_argument(\n        \"--max-rows\",\n        type=int,\n        default=None,\n        help=\"Optional row limit for development / memory-constrained runs\",\n    )\n    return parser.parse_args()"""), 1)
path.write_text(text)
