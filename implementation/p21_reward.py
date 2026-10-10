"""Two locked teacher sources share every input, pool, parser and reward gate."""
import json
import os
from pathlib import Path
from state import atomic_json

SEEDS = [90, 91, 92]
SOURCES = {"q": ("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5"),
           "m": ("lingshu-medical-mllm/Lingshu-7B", "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9")}


def validate(manifest):
    if manifest.get("p21_version") != 1 or manifest["rollout_seeds"] != SEEDS or manifest["primary_teacher"] != "q" or manifest["training_authorized_in_this_round"]:
        raise ValueError("P21 source, primary or complete scope differs")
    if (len(manifest["stream"]), len(manifest["probe"])) != (32, 32) or set(manifest["teachers"]) != set(SOURCES):
        raise ValueError("Incomplete paired teacher scope")
    settings = []
    for key, (repo, revision) in SOURCES.items():
        spec = manifest["teachers"][key]
        if (spec["repo"], spec["revision"]) != (repo, revision): raise ValueError("Teacher source changed")
        settings.append({k: v for k, v in spec.items() if k not in {"repo", "revision", "model_path"}})
    if settings[0] != settings[1] or settings[0].get("max_new_tokens") != 1024 or not settings[0].get("evidence_first") or not settings[0].get("json_suffix"):
        raise ValueError("Teacher prompt/decoding settings differ")
    if manifest["teacher"] != manifest["teachers"]["q"] or manifest["gate"] != {"minimum_accepted": 16, "minimum_correct": 12, "minimum_precision": .75, "wrong_positive_drop_pp": 10}:
        raise ValueError("Primary or reward thresholds differ")
    if set(manifest["configurations"]) != {f"pool-{arm}{seed}" for seed in SEEDS for arm in ["s", "t"]}:
        raise ValueError("Incomplete six-pool scope")
    for seed in SEEDS:
        for arm, sampler in [("s", (.7, .95)), ("t", (1., 1.))]:
            cfg = manifest["configurations"][f"pool-{arm}{seed}"]
            if cfg["seed"] != seed or cfg["method"] != "Frozen SC-8" or cfg["evaluation_labels_allowed"] or cfg.get("model_dtype", "bfloat16") != "bfloat16" or cfg["probability_check_bf16_atol"] != .1 or (cfg["temperature"], cfg["top_p"]) != sampler:
                raise ValueError("Frozen sampling or numerical guard differs")


def main():
    folder = Path(os.environ["P0_ROOT"]) / "outputs" / os.environ["P2_CAMPAIGN"]
    manifest = json.loads((folder / "manifest.json").read_text()); validate(manifest)
    mode = os.environ["P2_MODE"]
    if mode == "p21_resources":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("CPU resource stage has CUDA")
        for spec in manifest["teachers"].values():
            path = Path(spec["model_path"]); receipt = json.loads((path / "download_receipt.json").read_text())
            if receipt["repo"] != spec["repo"] or receipt["revision"] != spec["revision"] or not receipt["publisher_size_check"]: raise ValueError("Pinned warm resource receipt differs")
            for item in receipt["files"]:
                file = path / item["name"]
                if not file.resolve().is_relative_to(path.resolve()) or file.stat().st_size != item["bytes"]: raise ValueError("Warm publisher file size or path differs")
        atomic_json(folder / "resource_check.json", {"sealed": True, "teacher_keys": ["q", "m"], "publisher_size_check": True,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "gpu_process_seconds": 0, "full_file_hash_comparison": False})
    elif mode == "p21_verify":
        from p10_verifier import main as verify
        key = os.environ["P2_TEACHER_KEY"]
        if key not in SOURCES: raise ValueError("Unknown paired teacher")
        verify(key)
    elif mode == "p21_pool":
        import torch
        from p16_pool import main as collect
        from p20_pool import CachedCheckEngine
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        collect(CachedCheckEngine)
    elif mode == "p21_score":
        from p16_score import main as score
        from verified_reward import checked_targets
        digest = (folder / "manifest.sha256").read_text().strip()
        entries = manifest["stream"] + manifest["probe"]
        for key in SOURCES:
            signal = json.loads((folder / "reference" / key / "targets.json").read_text())
            checked_targets({**signal, "stream": signal["stream"] + signal["probe"]},
                {**manifest, "stream": entries, "teacher": manifest["teachers"][key]}, digest)
            if signal["code_commit"] != os.environ["P2_CODE_COMMIT"] or any(len(r["readouts"]) != 2 for r in signal["stream"] + signal["probe"]):
                raise ValueError("Both teacher sources must seal completely before scoring")
        # Both source audits must seal before publishing the fixed primary decision.
        results = {}
        for key in SOURCES:
            score("reference/" + key, "scores/" + key, manifest["teachers"][key])
            results[key] = json.loads((folder / "scores" / key / "qualification.json").read_text())
        atomic_json(folder / "scores/qualification.json", {"go": results["q"]["go"], "primary_teacher": "q",
            "teachers": results, "primary_selected_after_scoring": False, "training_performed": False})
    else: raise ValueError("Unknown P21 stage")


if __name__ == "__main__": main()
