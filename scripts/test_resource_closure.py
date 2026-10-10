"""A missing, failed, mismatched or prior-boot resource receipt must block GPU work."""
import os
os.environ.setdefault("P0_ROOT", ".")
os.environ.setdefault("P2_CAMPAIGN", "fixture")
from p10_pipeline import check_resource_closure


def main():
    ledger = {"status": "all_planned_jobs_completed", "jobs": [{"return_code": 0}]}
    spec = {"repo": "fixture", "revision": "pinned"}
    download = {**spec, "publisher_size_check": True}
    check_resource_closure(ledger, download, spec, "boot", "boot")
    bad_ledgers = [{"status": "all_planned_jobs_completed", "jobs": []},
                   {"status": "stopped_after_job_failure", "jobs": [{"return_code": 1}]},
                   {"status": "all_planned_jobs_completed", "jobs": [{"return_code": 0, "timeout_or_stop": True}]}]
    cases = [(item, download, "boot", "boot") for item in bad_ledgers]
    cases += [(ledger, {**download, "revision": "different"}, "boot", "boot"),
              (ledger, {**download, "publisher_size_check": False}, "boot", "boot"),
              (ledger, download, "boot", "new-boot"), (ledger, download, "", "")]
    for item, resource, expected, current in cases:
        try: check_resource_closure(item, resource, spec, expected, current)
        except ValueError: pass
        else: raise AssertionError("Invalid resource closure admitted a GPU stage")
    print("Resource completion and boot fences passed")


if __name__ == "__main__":
    main()
