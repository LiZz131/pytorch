#!/usr/bin/env bash
# Runs the GPU unit matrix (each case x mode in its own process) on a GPU pod.
#   run_unit.sh [case ...]            default: all cases
# Environment: CGMO_ROOT (default: the repo containing this tree) decides which tree is tested.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export CGMO_ROOT="${CGMO_ROOT:-$(cd "${HERE}/../.." && pwd)}"
export CGMO_TEST_OUT="${CGMO_TEST_OUT:-${HERE}/results}"
export LD_LIBRARY_PATH="${CGMO_ROOT}/cuda_graph_memopt/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${CGMO_ROOT}:${CGMO_ROOT}/python:${HERE}:${PYTHONPATH:-}"
mkdir -p "${CGMO_TEST_OUT}/unit/logs"
cd "${HERE}"

CASES=("$@")
[ ${#CASES[@]} -eq 0 ] && CASES=(chain scratch frag grow multi multigroup memcpy nondet strict verify dump)
MODES_FOR() {
  case "$1" in
    verify) echo "off fold both" ;;
    dump) echo "tensor fold both" ;;
    memcpy|nondet|strict) echo "off tensor fold both" ;;
    *) echo "off tensor fold both" ;;
  esac
}
fail=0
for c in "${CASES[@]}"; do
  for m in $(MODES_FOR "$c"); do
    log="${CGMO_TEST_OUT}/unit/logs/${c}_${m}.log"
    if python3 test_gpu_unit.py "$c" "$m" >"$log" 2>&1; then
      echo "PASS $c $m"
    else
      echo "FAIL $c $m   (see $log)"; tail -15 "$log"; fail=1
    fi
  done
done
exit $fail
