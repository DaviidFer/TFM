from __future__ import annotations

from pathlib import Path

from app.core.domain import PHASE1_SCOPE
from app.core.toolbox_manifest import TOOLBOX_STAGES, validate_phase1_toolbox


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report = validate_phase1_toolbox(root=root)

    print("=== Phase 1 Check ===")
    print(f"domain_name: {PHASE1_SCOPE.domain_name}")
    print(f"data_root: {PHASE1_SCOPE.data_root}")
    print(f"timeframe: {PHASE1_SCOPE.timeframe}")
    print(f"notebook_as_toolbox: {PHASE1_SCOPE.notebook_as_toolbox}")
    print(f"notebook_runtime_allowed: {PHASE1_SCOPE.notebook_runtime_allowed}")
    print("")

    print("Toolbox stages frozen:")
    for s in TOOLBOX_STAGES:
        print(f"- {s.stage_id}: {s.callable_name} [{s.file_path}]")

    print("")
    if report["missing_toolbox_files"]:
        print("Missing toolbox files:")
        for item in report["missing_toolbox_files"]:
            print(f"- {item}")
    else:
        print("Missing toolbox files: none")

    if report["missing_sample_assets"]:
        print("Missing sample assets:")
        for item in report["missing_sample_assets"]:
            print(f"- {item}")
    else:
        print("Missing sample assets: none")

    has_errors = bool(report["missing_toolbox_files"] or report["missing_sample_assets"])
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

