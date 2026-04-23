from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, List
from uuid import uuid4

from app.phase5_check import main as run_phase5_check
from app.tests.test_contracts_state_store import (
    test_promoted_spec_serialization_contains_lifecycle_value,
    test_state_store_roundtrip_for_state_metrics_events,
)
from app.tests.test_dashboard_snapshot import test_dashboard_snapshot_after_phase5_cycle
from app.tests.test_runtime_logs import test_runtime_structured_logs_include_key_events


@dataclass
class CheckResult:
    name: str
    ok: bool
    elapsed_sec: float
    error: str = ""


def _run_test(name: str, fn: Callable[[], None]) -> CheckResult:
    t0 = perf_counter()
    try:
        fn()
        return CheckResult(name=name, ok=True, elapsed_sec=perf_counter() - t0)
    except Exception as exc:  # pragma: no cover - utilitario de ejecución
        return CheckResult(name=name, ok=False, elapsed_sec=perf_counter() - t0, error=str(exc))


def _benchmark_phase5() -> CheckResult:
    t0 = perf_counter()
    bench_db = Path("app/.tmp/tests") / f"phase5_bench_{uuid4().hex[:8]}.sqlite"
    bench_db.parent.mkdir(parents=True, exist_ok=True)
    rc = int(run_phase5_check(db_path=bench_db))
    elapsed = perf_counter() - t0
    if rc != 0:
        return CheckResult(name="benchmark_phase5", ok=False, elapsed_sec=elapsed, error=f"exit_code={rc}")
    return CheckResult(name="benchmark_phase5", ok=True, elapsed_sec=elapsed)


def main() -> int:
    print("=== Phase 7 Check ===")
    print("Bateria: unitarios + integracion + benchmark")

    tests: List[tuple[str, Callable[[], None]]] = [
        ("unit_contracts_promoted_spec", test_promoted_spec_serialization_contains_lifecycle_value),
        ("unit_state_store_roundtrip", test_state_store_roundtrip_for_state_metrics_events),
        ("integration_dashboard_snapshot", test_dashboard_snapshot_after_phase5_cycle),
        ("integration_runtime_logs", test_runtime_structured_logs_include_key_events),
    ]

    results: List[CheckResult] = []
    for name, fn in tests:
        res = _run_test(name, fn)
        results.append(res)
        status = "OK" if res.ok else "FAIL"
        print(f"[{status}] {res.name} ({res.elapsed_sec:.2f}s)")
        if not res.ok:
            print(f"      error: {res.error}")

    bench = _benchmark_phase5()
    results.append(bench)
    print(f"[{'OK' if bench.ok else 'FAIL'}] {bench.name} ({bench.elapsed_sec:.2f}s)")
    if not bench.ok:
        print(f"      error: {bench.error}")
    else:
        # Umbral informativo para detectar degradación de rendimiento.
        target_sec = 35.0
        if bench.elapsed_sec > target_sec:
            print(f"[WARN] benchmark por encima de objetivo ({bench.elapsed_sec:.2f}s > {target_sec:.2f}s)")
        else:
            print(f"[OK] benchmark dentro de objetivo ({bench.elapsed_sec:.2f}s <= {target_sec:.2f}s)")

    failed = [r for r in results if not r.ok]
    print(f"tests_total: {len(results)}")
    print(f"tests_failed: {len(failed)}")
    if failed:
        raise RuntimeError("Phase 7 check failed.")

    print("Phase 7 check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

