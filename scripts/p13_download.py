"""Bounded native HTTPS resume through the required proxy, without visible URLs."""
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import quote
from p10_download import main


def transfer_file(url, path, size, proxy):
    def quoted(value):
        if "\n" in value or "\r" in value:
            raise ValueError("Newline in curl configuration")
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    for attempt in range(1, 4):
        before = path.stat().st_size if path.exists() else 0
        if before > size:
            raise ValueError("Partial resource exceeds publisher size")
        if before == size:
            return
        config = "\n".join(["location", "fail", "silent", "show-error", "http1.1", "connect-timeout = 30",
            "max-time = 900", 'continue-at = "-"', "proxy = " + quoted(proxy),
            "url = " + quoted(url), "output = " + quoted(str(path))]) + "\n"
        env = dict(os.environ, NO_PROXY="", no_proxy="")
        result = subprocess.run(["curl", "-q", "--config", "-"], input=config, text=True,
                                capture_output=True, env=env, timeout=930)
        after = path.stat().st_size if path.exists() else 0
        print(json.dumps({"stage": "resource_transfer", "attempt": attempt, "before_bytes": before,
                          "after_bytes": after, "expected_bytes": size, "return_code": result.returncode}), flush=True)
        if after < before or after > size:
            raise ValueError("Resumed transfer changed or exceeded the retained byte range")
        if result.returncode == 0:
            if after != size:
                raise ValueError("Completed transfer size differs from publisher")
            return
        if attempt < 3:
            time.sleep(2)
    raise RuntimeError("Three bounded native HTTPS attempts failed; partial resource retained")


def download(repo, revision, destination, files):
    for item in files:
        name = item["rfilename"]
        target = destination / name
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Publisher filename escapes allocated storage")
        if target.exists():
            continue  # The shared exporter checks every retained file against publisher size.
        target.parent.mkdir(parents=True, exist_ok=True)
        etag = item.get("lfs", {}).get("sha256")
        if etag is not None and (len(etag) != 64 or any(c not in "0123456789abcdef" for c in etag)):
            raise ValueError("Publisher partial-file key is invalid")
        partials = list((destination / ".cache/huggingface/download").glob(f"*.{etag}.incomplete")) if etag else []
        if len(partials) > 1:
            raise ValueError("Ambiguous retained Hugging Face partial resource")
        partial = partials[0] if partials else target.with_name(target.name + ".part")
        if not partial.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Retained partial escapes allocated storage")
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{quote(name, safe='/')}"
        transfer_file(url, partial, item["size"], os.environ["P10_PROXY"])
        os.replace(partial, target)


if __name__ == "__main__":
    main(download)
