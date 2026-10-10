"""Pinned public teacher resources; no CUDA context or medical uploads."""
import json
import hashlib
import os
import urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    root = Path(os.environ["P10_SOURCE_ROOT"])
    folder = Path(os.environ["P0_ROOT"]) / "outputs" / os.environ["P2_CAMPAIGN"]
    raw = (folder / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != (folder / "manifest.sha256").read_text().strip():
        raise ValueError("Locked download specification differs")
    spec = json.loads(raw)["teacher"]
    repo, revision = spec["repo"], spec["revision"]
    destination = Path(spec["model_path"])
    if destination.resolve() != (root / "models" / repo.split("/")[-1]).resolve():
        raise ValueError("Teacher download escapes allocated storage")
    receipt = destination / "download_receipt.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record["revision"] != revision or any((destination / p["name"]).stat().st_size != p["bytes"] for p in record["files"]):
            raise ValueError("Existing teacher resource receipt differs")
        print(json.dumps({"status": "already_complete", "revision": revision}), flush=True)
        return
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", timeout=30) as response:
        metadata = json.load(response)
    if metadata["sha"] != revision:
        raise ValueError("Publisher revision differs")
    allowed = [x for x in metadata["siblings"] if x["rfilename"].endswith((".json", ".txt", ".safetensors"))]
    snapshot_download(repo_id=repo, revision=revision, local_dir=destination,
                      allow_patterns=[x["rfilename"] for x in allowed], max_workers=2)
    files = []
    for item in allowed:
        size = (destination / item["rfilename"]).stat().st_size
        if size != item["size"]:
            raise ValueError("Downloaded resource byte size differs from publisher")
        files.append({"name": item["rfilename"], "bytes": size})
    record = {"repo": repo, "revision": revision, "files": files, "publisher_size_check": True,
              "full_file_hash_comparison": False, "medical_data_uploaded": False}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "complete", "revision": revision, "bytes": sum(x["bytes"] for x in files)}), flush=True)


if __name__ == "__main__":
    main()
