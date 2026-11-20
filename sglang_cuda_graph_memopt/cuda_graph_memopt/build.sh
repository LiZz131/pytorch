#!/usr/bin/env bash
# Build the native parts of cuda_graph_memopt into cuda_graph_memopt/lib (or $CGMO_LIB_DIR).
#
#   build.sh c10     patched libc10_cuda.so  (tensor feature)   ~2 min, needs the patched PyTorch c10/cuda sources
#                    (PYTORCH_SRC, default <repo>/third_party/pytorch or /workspace/.home/third_party/pytorch;
#                     fork https://github.com/LiZz131/pytorch branch feature/stream-mutex-groups, which carries
#                     patches/pytorch-c10-tmo.patch)
#   build.sh fold    graph_fold.so           (fold feature)     ~1 min, needs nvcc
#   build.sh all     both (default)
#
# Neither replaces anything in site-packages. libc10_cuda.so is picked up through
#   export LD_LIBRARY_PATH=<repo>/cuda_graph_memopt/lib:$LD_LIBRARY_PATH
# which must be set before python starts (the dynamic loader resolves libc10_cuda.so for torch).
set -euo pipefail

PKG="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${PKG}/.." && pwd)"
OUT="${CGMO_LIB_DIR:-${PKG}/lib}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
WHAT="${1:-all}"
mkdir -p "${OUT}"

build_c10() {
  local SRC="${PYTORCH_SRC:-}" WORK="${OUT}/build_c10"
  if [ -z "${SRC}" ]; then
    for cand in "${ROOT}/third_party/pytorch" "${ROOT}/3rdparty/pytorch" "/workspace/.home/third_party/pytorch"; do
      [ -d "${cand}/c10/cuda" ] && { SRC="${cand}"; break; }
    done
  fi
  [ -d "${SRC}/c10/cuda" ] || { echo "missing PyTorch sources (set PYTORCH_SRC; see patches/pytorch-c10-tmo.patch)"; exit 1; }
  grep -q tmo_set_capture_plan "${SRC}/c10/cuda/CUDACachingAllocator.cpp" || {
    echo "${SRC} is not patched: git -C ${SRC} apply -p1 ${PKG}/patches/pytorch-c10-tmo.patch"; exit 1; }
  echo "c10 sources: ${SRC}"
  mkdir -p "${WORK}"
  local PY_ROOT PY_LIB PY_INC
  PY_ROOT="$(python3 -c 'import os,torch; print(os.path.dirname(torch.__file__))')"
  PY_LIB="${PY_ROOT}/lib"
  PY_INC="${PY_ROOT}/include"
  local CXXFLAGS=(-std=c++17 -O2 -DNDEBUG -fPIC -fvisibility=hidden
    -DC10_CUDA_BUILD_MAIN_LIB -DC10_CUDA_NO_CMAKE_CONFIGURE_FILE -DPYTORCH_C10_DRIVER_API_SUPPORTED
    -I "${SRC}" -I "${PY_INC}" -I "${CUDA_HOME}/include")
  local SRCS=(CUDAAllocatorConfig.cpp CUDACachingAllocator.cpp CUDADeviceAssertionHost.cpp
    CUDAException.cpp CUDAFunctions.cpp CUDAMallocAsyncAllocator.cpp CUDAMiscFunctions.cpp
    CUDAStream.cpp impl/CUDAGuardImpl.cpp impl/CUDATest.cpp driver_api.cpp)
  local OBJS=() src obj
  for src in "${SRCS[@]}"; do
    obj="${WORK}/$(basename "${src}" .cpp).o"
    echo "cc ${src}"
    g++ "${CXXFLAGS[@]}" -c "${SRC}/c10/cuda/${src}" -o "${obj}" &
    OBJS+=("${obj}")
  done
  wait
  g++ -shared -o "${OUT}/libc10_cuda.so" "${OBJS[@]}" \
    -L "${PY_LIB}" -lc10 -L "${CUDA_HOME}/lib64" -lcudart -ldl \
    -Wl,-rpath,"${PY_LIB}" -Wl,-rpath,"${CUDA_HOME}/lib64"
  echo "built ${OUT}/libc10_cuda.so"
  nm -D "${OUT}/libc10_cuda.so" | grep -c ' T tmo_' | xargs echo "exported tmo_* symbols:"
}

build_fold() {
  local SRC="${PKG}/fold"
  local C=(nvcc -O3 -std=c++17 -arch=native -I "${SRC}" -Xcompiler -fPIC)
  "${C[@]}" -rdc=true -c "${SRC}/graph_fold.cu" -o "${OUT}/graph_fold.o"
  "${C[@]}" -dlink "${OUT}/graph_fold.o" -o "${OUT}/graph_fold_dlink.o" -lcudadevrt
  "${C[@]}" -shared "${OUT}/graph_fold.o" "${OUT}/graph_fold_dlink.o" -o "${OUT}/graph_fold.so" -lcudadevrt -lcuda
  echo "built ${OUT}/graph_fold.so"
}

case "${WHAT}" in
  c10) build_c10 ;;
  fold) build_fold ;;
  all) build_c10 & build_fold; wait ;;
  *) echo "usage: $0 [c10|fold|all]"; exit 2 ;;
esac
