"""Download both Zenodo datasets into data/raw/ (never committed to git).

File names are discovered through the Zenodo record API, not hard-coded.
Each file is checksum-verified when Zenodo supplies an md5. Archives are
extracted next to the download. The record metadata (title, DOI, license) is
saved so the README can credit the datasets correctly.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import requests

try:  # works as `python -m src.download` and as plain import in the notebook
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore


def _get(url: str, **kw) -> requests.Response:
    last = None
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:  # retry with backoff
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last}")


def get_record(record_id: str) -> dict:
    return _get(f"{config.ZENODO_BASE}/api/records/{record_id}").json()


def _file_entries(record: dict) -> list[dict]:
    out = []
    for f in record.get("files", []):
        name = f.get("key") or f.get("filename")
        url = (f.get("links") or {}).get("self") or (f.get("links") or {}).get("download")
        if not url:
            url = f"{config.ZENODO_BASE}/records/{record['id']}/files/{name}?download=1"
        out.append({"name": name, "size": f.get("size"), "checksum": f.get("checksum"), "url": url})
    return out


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(zpath: Path, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    with zipfile.ZipFile(zpath) as z:
        for member in z.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"Unsafe path in zip: {member.filename}")
        z.extractall(dest)
        names = z.namelist()
    return names


def download_record(key: str, force: bool = False) -> Path:
    meta = config.DATASETS[key]
    rid = meta["record_id"]
    dest = config.DATA_RAW / key
    dest.mkdir(parents=True, exist_ok=True)

    record = get_record(rid)
    (dest / "record_metadata.json").write_text(json.dumps(record, indent=2))
    md = record.get("metadata", {})
    print(f"[{key}] {md.get('title')}")
    print(f"[{key}] DOI: {record.get('doi') or md.get('doi')} | "
          f"license: {(md.get('license') or {}).get('id')} | "
          f"version: {md.get('version')}")

    for f in _file_entries(record):
        path = dest / f["name"]
        if path.exists() and not force:
            print(f"[{key}] already have {f['name']}")
        else:
            print(f"[{key}] downloading {f['name']} ({f['size']} bytes)")
            with _get(f["url"], stream=True) as r, open(path, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        if f["checksum"] and f["checksum"].startswith("md5:"):
            ok = _md5(path) == f["checksum"].split(":", 1)[1]
            print(f"[{key}] md5 {'OK' if ok else 'MISMATCH'}: {f['name']}")
            if not ok:
                raise RuntimeError(f"Checksum mismatch for {f['name']}")
        if path.suffix.lower() == ".zip":
            names = _safe_extract(path, dest / "extracted")
            print(f"[{key}] extracted {len(names)} entries into {dest / 'extracted'}")
    return dest


def main() -> None:
    for key in config.DATASETS:
        download_record(key)
    print("Done. Raw data is in", config.DATA_RAW)


if __name__ == "__main__":
    sys.exit(main())
