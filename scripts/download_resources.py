"""Pinned official downloads; rerunning resumes HF partial files in place."""
import json
import os
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(os.environ["P0_ROOT"])
JOBS = {
    "slake": ("BoKelvin/SLAKE", "dataset", "data/SLAKE/raw", None),
    "medxpert": ("TsinghuaC3I/MedXpertQA", "dataset", "data/MedXpertQA-MM/raw", ["MM/*", "images.zip", "README.md", ".gitattributes"]),
    "model": ("Qwen/Qwen2.5-VL-3B-Instruct", "model", "models/Qwen2.5-VL-3B-Instruct", None),
}


def prepare(key):
    repo, kind, relative, patterns = JOBS[key]
    api = json.loads((ROOT / "metadata/pinned_downloads.json").read_text())[key]
    destination = ROOT / relative
    result = {"repo": repo, "revision": api["sha"], "path": str(destination), "status": "downloading"}
    status_path = ROOT / "metadata" / f"{key}_download.json"
    status_path.write_text(json.dumps(result, indent=2))
    try:
        snapshot_download(repo_id=repo, repo_type=kind, revision=api["sha"], local_dir=destination, allow_patterns=patterns, max_workers=2)
        # One low-cost post-download check against publisher-reported byte sizes.
        files = []
        for item in api["siblings"]:
            name = item["rfilename"]
            if key == "medxpert" and name.startswith("Text/"):
                continue
            p = destination / name
            assert p.is_file(), f"Missing {p}"
            assert p.stat().st_size == item["size"], f"Unexpected size: {p}"
            files.append({"file": name, "bytes": p.stat().st_size})
        result.update(status="download_complete", files=files, total_bytes=sum(f["bytes"] for f in files))
        status_path.write_text(json.dumps(result, indent=2))
        if kind == "dataset":
            expanded = destination.parent / "extracted"
            expanded.mkdir(exist_ok=True)
            for archive in destination.glob("*.zip"):
                marker = expanded / ("." + archive.name + ".complete")
                if marker.exists():
                    continue
                with zipfile.ZipFile(archive) as z:
                    for member in z.infolist():
                        target = (expanded / member.filename).resolve()
                        if not target.is_relative_to(expanded.resolve()):
                            raise ValueError("Unsafe archive path")
                    z.extractall(expanded)  # ZIP's built-in CRC validation is retained.
                    marker.write_text(json.dumps({"archive": str(archive), "entries": len(z.infolist())}))
                print("EXTRACTED", key, archive.name, flush=True)
            result["extracted_path"] = str(expanded)
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    status_path.write_text(json.dumps(result, indent=2))
    print(key, result["status"], flush=True)
    return result["status"] == "complete"


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        ok = list(pool.map(prepare, JOBS))
    raise SystemExit(0 if all(ok) else 1)
