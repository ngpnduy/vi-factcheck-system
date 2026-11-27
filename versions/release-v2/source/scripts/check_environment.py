"""Check the recorded research environment without installing or upgrading packages."""
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    expected = dict(line.strip().split('==', 1) for line in
                    (root / 'requirements-tested.txt').read_text().splitlines()
                    if line.strip() and not line.startswith('#'))
    actual, errors = {}, []
    for package, version in expected.items():
        try:
            actual[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual[package] = None
        if actual[package] != version:
            errors.append(f'{package}: expected {version}, found {actual[package]}')
    sys.path.insert(0, str(root / 'src'))
    import vifactcheck
    package_path = Path(vifactcheck.__file__).resolve()
    if not package_path.is_relative_to(root / 'src' / 'vifactcheck'):
        errors.append(f'Wrong vifactcheck package: {package_path}')
    print(json.dumps({'python': sys.version, 'validated_python': '3.12', 'packages': actual,
                      'package_path': str(package_path), 'errors': errors}, indent=2))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
