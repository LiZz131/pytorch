#!/usr/bin/env bash
# Green Context matrix (A = 36 SM optimised stream, B = 96 SM peer) on a GPU pod.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export CGMO_ROOT="${CGMO_ROOT:-$(cd "${HERE}/../.." && pwd)}"
export CGMO_TEST_OUT="${CGMO_TEST_OUT:-${HERE}/results}"
export LD_LIBRARY_PATH="${CGMO_ROOT}/cuda_graph_memopt/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${CGMO_ROOT}:${CGMO_ROOT}/python:${HERE}:${PYTHONPATH:-}"
mkdir -p "${CGMO_TEST_OUT}/green/logs"; cd "${HERE}"
fail=0
for m in off tensor fold both; do
  if python3 test_green.py "$m" > "${CGMO_TEST_OUT}/green/logs/green_$m.log" 2>&1; then echo "PASS green $m"; else echo "FAIL green $m"; tail -20 "${CGMO_TEST_OUT}/green/logs/green_$m.log"; fail=1; fi
done
python3 - <<'PY'
import json, os
print("mode  bitwise  sm_ids(A)  sm_ids(from B)  only_a  only_b  both   hidden  event_overlap")
for m in ("off", "tensor", "fold", "both"):
    r = json.load(open(os.path.join(os.environ["CGMO_TEST_OUT"], "green", f"green_{m}.json")))
    print(f'{m:6s} {r["bitwise_vs_eager"]!s:6s} {r["sm_ids_replay_on_A"]["count"]:4d} {r["sm_ids_replay_launched_from_B"]["count"]:6d}'
          f' {r["only_a_ms"]:8.2f} {r["only_b_ms"]:8.2f} {r["both_ms"]:8.2f} {r["hidden_fraction_of_A"]:6.3f} {r["event_overlap_fraction_of_A"]:6.3f}  {r.get("fold_status")} {r.get("tensor_mode")}')
PY
exit $fail
