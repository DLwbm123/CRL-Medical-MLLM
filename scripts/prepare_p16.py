"""Freeze a label-blind, training-only medical reward validation campaign."""
import copy
import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from PIL import Image
from prepare_campaign import write_manifest
from state import atomic_json

META_KEYS = {"qid", "img_name", "q_lang", "answer_type", "question", "modality", "content_type"}
YES_NO = re.compile(r"^(is|are|does|do|has|have|can|was|were)\b", re.I)
SEEDS = [75, 76, 77]


def select_rows(rows, excluded, seed=20261010):
    if any(set(r) != META_KEYS for r in rows):
        raise ValueError("Selection accepts exact label-free metadata only")
    groups = {}
    for r in rows:
        if r["q_lang"] == "en" and r["answer_type"] == "CLOSED" and r["content_type"] != "KG" and YES_NO.match(r["question"].strip()) and r["img_name"] not in excluded:
            groups.setdefault(r["img_name"], []).append(r)
    rng = random.Random(seed)
    selected = [rng.choice(sorted(v, key=lambda r: r["qid"])) for _, v in sorted(groups.items())]
    rng.shuffle(selected)
    if len(selected) < 64:
        raise ValueError("Fewer than64 eligible disjoint image groups")
    return selected, len(groups)


def binary_domain_metadata(rows):
    """Use annotation vocabulary only to define the new binary task, not correctness."""
    if any(not isinstance(r.get("answer"), str) for r in rows):
        raise ValueError("Training annotation has no string answer")
    return [{k: r[k] for k in META_KEYS} for r in rows if r["answer"].strip().lower() in {"yes", "no"}]


def validate_binary_domain(path, selected_ids):
    seen = set()
    for line in path.open():
        r = json.loads(line)
        if r["id"] not in selected_ids: continue
        if r["id"] in seen or not isinstance(r["answer"], str) or r["answer"].strip().lower() not in {"yes", "no"}:
            raise ValueError("Selected scoring-view annotation violates the binary domain")
        seen.add(r["id"])
    if seen != set(selected_ids): raise ValueError("Selected scoring-view annotations are incomplete")
    return {"admitted": True, "groups": len(seen), "annotation_domain_only": True, "class_labels_exported": False}


def heldout_image_references(path):
    # The exclusion reader extracts only image-reference arrays, never questions or labels.
    result = set()
    for line in path.open():
        match = re.search(r'"images"\s*:\s*(\[[^\]]*\])', line)
        if not match:
            raise ValueError("Held-out identity view lacks image references")
        result.update(json.loads(match.group(1)))
    return result


def thumbnail(path):
    with Image.open(path) as image:
        return image.convert("RGB").resize((32, 32)).tobytes()


def main():
    root = Path(os.environ["P0_ROOT"]); source = Path(os.environ["P10_SOURCE_ROOT"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    recovery = os.environ.get("P2_MODE") == "p17_prepare"
    previous_name = "p16-20261010T105716" if recovery else "p15-20261010T095303"
    previous = source / "workspaces" / previous_name / "outputs" / previous_name
    prefix = "p17" if recovery else "p16"; seeds = [78, 79, 80] if recovery else SEEDS
    end = json.loads((previous / "FINAL.json").read_text())
    delivery = json.loads((previous / "public_delivery_receipt.json").read_text())
    closure = json.loads((previous / "completion_audit.json").read_text())
    expected_status = "failed" if recovery else "reward_qualification_negative"
    if end["status"] != expected_status or not closure["owned_processes_ended"] or not delivery["remote_sha_verified"] or not delivery["anonymous_access_verified"]:
        raise ValueError("Previous result must close and be publicly delivered")
    if recovery and json.loads((previous / "offline_failure_audit.json").read_text())["failure_type"] != "input_label_contract_failure":
        raise ValueError("Recovery requires the documented input-domain failure")
    if folder.exists():
        raise FileExistsError("No duplicate data preparation")
    folder.mkdir(parents=True)
    for name in ["inputs", "reference", "scores", "runs", "controllers"]: (folder / name).mkdir()
    receipt = json.loads((source / "metadata/slake_download.json").read_text())
    if receipt["status"] != "complete" or receipt["revision"] != "a9083ce6c34ac3ffb17671a605962924d8a8f9e9":
        raise ValueError("Pinned SLAKE resources differ")
    raw = json.loads((source / "data/SLAKE/raw/train.json").read_text())
    metadata = binary_domain_metadata(raw) if recovery else [{k: r[k] for k in META_KEYS} for r in raw]
    del raw
    excluded = set.union(*(heldout_image_references(source / "views/adaptation_inputs/SLAKE" / (split + ".jsonl")) for split in ["validation", "test"]))
    heldout_exclusions = len(excluded)
    old_manifest = json.loads((previous / "manifest.json").read_text())
    if recovery: excluded.update(image for e in old_manifest["stream"] + old_manifest["probe"] for image in e["images"])
    selected, eligible = select_rows(metadata, excluded)
    old_prints = set()
    old_folders = [previous]
    if recovery: old_folders.append(source / "workspaces/p15-20261010T095303/outputs/p15-20261010T095303")
    for old in old_folders:
        old_scope = json.loads((old / "manifest.json").read_text())
        for entry in old_scope["stream"] + old_scope["probe"]:
            row = json.loads((old / entry["input"]).read_text())
            old_prints.update(thumbnail(Path(p)) for p in row["image_paths"])
    seen = set(old_prints); clean = []
    image_root = (source / "data/SLAKE/extracted/imgs").resolve()
    duplicate_groups = 0
    for r in selected:
        path = (image_root / r["img_name"]).resolve()
        if not path.is_relative_to(image_root): raise ValueError("Image path escapes source")
        fingerprint = thumbnail(path)
        if fingerprint in seen:
            duplicate_groups += 1; continue
        seen.add(fingerprint); clean.append((r, path))
        if len(clean) == 64: break
    if len(clean) != 64: raise ValueError("Insufficient groups after thumbnail duplicate exclusions")
    entries = []
    for index, (r, path) in enumerate(clean):
        sample_id = f"SLAKE/train/{r['qid']}"
        row = dict(id=sample_id, source_id=r["qid"], dataset="SLAKE", split="train", question=r["question"],
                   options={"A": "yes", "B": "no"}, images=[r["img_name"]], image_paths=[str(path)], language="en")
        relative = f"inputs/i{index:03d}.json"; atomic_json(folder / relative, row)
        entries.append({"id": sample_id, "split": "train", "group": f"g{index:03d}", "input": relative, "images": row["images"]})
    teacher = copy.deepcopy(old_manifest["teacher"])
    configs = {}
    base = json.loads((source / "configs/p2_c.json").read_text())
    for seed in seeds:
        for arm, temperature, top_p in [("s", .7, .95), ("t", 1., 1.)]:
            cfg = {**base, "method": "Frozen SC-8", "seed": seed, "gpu_index": 0, "temperature": temperature, "top_p": top_p, "reward_source": "majority"}
            name = f"pool-{arm}{seed}"; configs[name] = cfg
            atomic_json(root / "configs" / f"{prefix}_{arm}{seed}.json", cfg)
    manifest = {prefix + "_version": 1, "stream": entries[:32], "probe": entries[32:], "rollout_seeds": seeds,
        "teacher": teacher, "configurations": configs, "source_revision": receipt["revision"],
        "selection_seed": 20261010, "selection_receives_labels": recovery, "annotation_domain_only": recovery, "correctness_used_for_selection": False,
        "heldout_access": "image-reference metadata solely for exclusions; no labels or model evaluation",
        "patient_disjointness_verified": False, "grouping": "unique image references and conservative32x32 RGB thumbnail exclusions",
        "broad_content_or_patient_duplicate_equivalence_verified": False, "pretraining_overlap_excluded": False,
        "training_authorized_in_this_round": False, "selected_n": 64,
        "gate": {"minimum_accepted": 16, "minimum_correct": 12, "minimum_precision": .75, "wrong_positive_drop_pp": 10}}
    digest = write_manifest(folder / "manifest.json", manifest)
    if recovery:
        domain = validate_binary_domain(source / "views/evaluation_labels/SLAKE/train.jsonl", {e["id"] for e in entries})
        atomic_json(folder / "binary_domain_admission.json", {**domain, "manifest_sha256": digest})
    now = datetime.now(timezone.utc); estimate = 6 * 4 * 4590.984604918864 + 7200
    budget = {"total_gpu_process_seconds": 259200., "prior_gpu_process_seconds": end["cumulative_gpu_process_seconds"],
        "remaining_total_gpu_process_seconds_at_start": end["remaining_gpu_process_seconds"], "budget_reset": False,
        "gpu_stop_utc": (now + timedelta(hours=46)).isoformat(), "hard_deadline_utc": (now + timedelta(hours=48)).isoformat(),
        "recovery_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "complete_scope_estimated_gpu_process_seconds": estimate, "complete_scope_required_with_reserve": estimate * 1.2,
        "reserve_fraction": .2, "qualified_training_pipeline_enabled": False, "authorized_physical_gpus": [0, 1]}
    if estimate * 1.2 > budget["remaining_total_gpu_process_seconds_at_start"] or estimate * 1.2 > 46 * 3600:
        raise ValueError("Whole qualification scope plus20% cannot fit")
    atomic_json(root / "metadata/campaign_budget.json", budget)
    atomic_json(folder / "resources-plan.json", {"jobs": [{"label": "resource", "gpu": False, "environment": {"P2_MODE": "download_transport"}, "max_seconds": 7200}]})
    atomic_json(folder / "qualification-plan.json", {"minimum_free_mib": 24000, "jobs": [{"label": "frozen-verifier", "environment": {"P2_MODE": "verify"}, "max_seconds": 7200}]})
    atomic_json(folder / "main-plan.json", {"minimum_free_mib": 24000, "jobs": [
        {"label": name, "environment": {"P2_MODE": "p16_pool", "P2_RUN": name}, "max_seconds": 21600, "expected_cursor": 64} for name in configs]})
    audit = {"admitted": True, "training_admitted": False, "data_authorized_by_direct_user_delegation": True,
        "source": "BoKelvin/SLAKE pinned official re-cleaned1.0", "split": "train only", "new_groups": 64,
        "calibration_groups": 32, "verification_groups": 32, "eligible_image_groups": eligible,
        "heldout_image_refs_excluded": heldout_exclusions, "previous_image_refs_excluded": 64 if recovery else 0, "thumbnail_duplicate_groups_skipped": duplicate_groups,
        "selection_receives_labels": recovery, "annotation_domain_only": recovery, "correctness_used_for_selection": False, "no_final_test_generation_or_scoring": True,
        "patient_disjointness_verified": False, "pretraining_overlap_excluded": False,
        "modalities": dict(Counter(r["modality"] for r, _ in clean)), "budget": budget}
    atomic_json(folder / "budget_admission.json", audit); atomic_json(folder / "preparation.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__": main()
