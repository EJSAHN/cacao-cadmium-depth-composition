from __future__ import annotations
import hashlib, json, os, shutil, zipfile
from pathlib import Path

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def preserve_source(source, destination, expected_hash):
    """Never replace a different existing input, including a corrupted snapshot."""
    source, destination = Path(source), Path(destination)
    if sha256(source) != expected_hash:
        raise ValueError('Bundled source checksum mismatch. Re-download the CODE6 package.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != expected_hash:
            raise ValueError('Existing source differs; it was preserved: ' + str(destination))
    else:
        temp = destination.with_suffix('.copying')
        shutil.copyfile(source, temp)
        if sha256(temp) != expected_hash:
            raise ValueError('Source copy failed integrity validation: ' + str(temp))
        temp.replace(destination)
    return destination


def collect(run_dir, root, run_id, failed=False):
    exports = root / 'exports'
    exports.mkdir(parents=True, exist_ok=True)
    name = run_id + ('_FAILED_REVIEW.zip' if failed else '_REVIEW.zip')
    target = exports / name
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(run_dir.rglob('*')):
            if p.is_file() and not p.name.endswith('.tmp'):
                z.write(p, p.relative_to(run_dir).as_posix())
    preferred = target
    downloads = Path(os.environ.get('USERPROFILE', str(Path.home()))) / 'Downloads'
    if os.name == 'nt' and downloads.is_dir():
        try:
            preferred = downloads / name
            shutil.copy2(target, preferred)
        except OSError as exc:
            print('Downloads copy unavailable; use the project export: ' + str(exc))
            preferred = target
    print('PROJECT_REVIEW_ZIP=' + str(target))
    print('UPLOAD_THIS_ZIP=' + str(preferred))
    return target


