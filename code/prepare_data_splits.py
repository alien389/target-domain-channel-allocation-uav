"""Materialize recorded JSON/YAML path templates into NEW files only."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_paths import expand

def convert(value):
    if isinstance(value, dict):
        return {expand(k) if isinstance(k, str) else k: convert(v) for k, v in value.items()}
    if isinstance(value, list):
        return [convert(v) for v in value]
    return expand(value) if isinstance(value, str) else value

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing overwrite: {args.output}")
    if not args.template.is_file():
        raise FileNotFoundError(args.template)
    if args.template.suffix == '.json':
        obj = json.loads(args.template.read_text('utf-8-sig'))
        text = json.dumps(convert(obj), ensure_ascii=False, indent=2) + '\n'
    elif args.template.suffix in ('.yaml', '.yml'):
        import yaml
        obj = yaml.safe_load(args.template.read_text('utf-8-sig'))
        text = yaml.safe_dump(convert(obj), sort_keys=False)
    else:
        parser.error('Only JSON/YAML templates are supported')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as fh:
        fh.write(text)
    print(f'Materialized new config: {args.output}')

if __name__ == '__main__':
    main()
