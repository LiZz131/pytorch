"""Static tensor placement for one CUDA-graph capture.

The allocator trace of a capture is a sequence of ``alloc`` and
``free_completed`` events on one stream. Each allocation is a buffer with a
fixed size and a half-open lifetime ``[start, end)``. A buffer that is still
allocated when the capture ends has ``end is None``: its address is baked into
the graph outputs and stays reserved for the graph's lifetime.

Placement gives every buffer one offset for its whole lifetime. Two buffers
whose lifetimes overlap must not overlap in address. This is the classic
dynamic storage-allocation problem (TVM's Unified Static Memory Planner calls
the same thing buffer-offset assignment). It is NP-hard, so the planner uses
TVM's default heuristic, greedy-by-size:

    https://github.com/apache/tvm/commit/756debc4488e580e1d3aca20bc3e95f397169cb1

Greedy-by-size places the largest buffer first, at the lowest offset that does
not collide with an already placed buffer of overlapping lifetime. A large
buffer therefore occupies a contiguous span before any smaller buffer is
allowed to split that span. That is the opposite of the caching allocator's
online first-fit, which will split a just-freed large block to satisfy a small
request and then cudaMalloc a second large block when the large request comes
back.
"""

from __future__ import annotations

# PyTorch's caching allocator rounds every request up to at least 512 bytes
# when roundup_power2_divisions is left at its default of 0.
# See c10/cuda/CUDACachingAllocator.cpp, kMinBlockSize and round_size.
MIN_BLOCK = 512

# Segment growth rules from get_allocation_size in the same file.
SMALL_SIZE = 1 << 20
SMALL_BUFFER = 2 << 20
MIN_LARGE_ALLOC = 10 << 20
ROUND_LARGE = 2 << 20
LARGE_BUFFER = 20 << 20

# Surviving buffers up to max(TOP_ZONE_MIN, total / TOP_ZONE_FRACTION) bytes go to the top zone.
TOP_ZONE_MIN = 256 << 10
TOP_ZONE_FRACTION = 256


def round_size(nbytes: int) -> int:
    if nbytes <= 0:
        raise ValueError(f"allocation size must be positive, got {nbytes}")
    if nbytes < MIN_BLOCK:
        return MIN_BLOCK
    return MIN_BLOCK * ((nbytes + MIN_BLOCK - 1) // MIN_BLOCK)


def segment_size(nbytes: int) -> int:
    """Bytes cudaMalloc will reserve for a single request of ``nbytes``."""
    size = round_size(nbytes)
    if size <= SMALL_SIZE:
        return SMALL_BUFFER
    if size < MIN_LARGE_ALLOC:
        return LARGE_BUFFER
    return ROUND_LARGE * ((size + ROUND_LARGE - 1) // ROUND_LARGE)


def arena_size(packed: int) -> int:
    """Smallest size >= ``packed`` that is its own segment size.

    The allocator picks the pool from the *requested* size and then rounds with
    ``get_allocation_size``. The arena is requested with ``arena_size`` bytes, so that size must
    map to itself: e.g. a 100 KiB span gives a 2 MiB small-pool segment, but a 2 MiB request
    goes to the large pool and would reserve 20 MiB. Iterate until stable (at most 3 steps).
    """
    size = segment_size(packed)
    for _ in range(4):
        nxt = segment_size(size)
        if nxt == size:
            return size
        size = nxt
    raise RuntimeError(f"arena size does not converge for {packed}")


class Buffer:
    __slots__ = ("index", "size", "start", "end", "offset")

    def __init__(self, index: int, size: int, start: int, end: int | None):
        self.index = index
        self.size = size
        self.start = start
        self.end = end
        self.offset = 0

    def overlaps(self, other: "Buffer") -> bool:
        return _intervals_overlap(self.start, self.end, other.start, other.end)


def _intervals_overlap(a0: int, a1: int | None, b0: int, b1: int | None) -> bool:
    # None means the buffer is still live at the end of the capture.
    a1e = a1 if a1 is not None else 1 << 62
    b1e = b1 if b1 is not None else 1 << 62
    return a0 < b1e and b0 < a1e


def buffers_from_trace(events: list[dict]) -> list[Buffer]:
    """Build buffers from allocator trace events on one stream.

    ``free_completed`` ends a lifetime. ``free_requested`` does not: during
    capture a block recorded on another stream stays unavailable until the
    deferred free completes, and reusing it earlier would alias a live address.
    """
    live: dict[int, Buffer] = {}
    buffers: list[Buffer] = []
    tick = 0
    for event in events:
        action = event["action"]
        if action == "alloc":
            buf = Buffer(
                index=len(buffers),
                size=round_size(int(event["size"])),
                start=tick,
                end=None,
            )
            buffers.append(buf)
            addr = int(event["addr"])
            if addr in live:
                raise RuntimeError(
                    f"address {addr:#x} allocated while a previous buffer is still live"
                )
            live[addr] = buf
            tick += 1
        elif action == "free_completed":
            addr = int(event["addr"])
            buf = live.pop(addr, None)
            if buf is not None:
                buf.end = tick
                tick += 1
    return buffers


def peak_live_bytes(buffers: list[Buffer]) -> int:
    events: list[tuple[int, int]] = []
    for buf in buffers:
        events.append((buf.start, buf.size))
        if buf.end is not None:
            events.append((buf.end, -buf.size))
    events.sort(key=lambda item: (item[0], item[1] > 0))
    live = 0
    peak = 0
    for _, delta in events:
        live += delta
        if live > peak:
            peak = live
    return peak


def fixed_buffer(index: int, offset: int, size: int) -> Buffer:
    """An obstacle that is already placed and lives for the whole capture (an earlier graph's
    output, or anything else that currently occupies the arena)."""
    buf = Buffer(-1 - index, size, 0, None)
    buf.offset = offset
    return buf


def greedy_by_size(buffers: list[Buffer], fixed: list[Buffer] = (), ceiling: int | None = None) -> int:
    """Assign ``buffer.offset``. Return the exclusive end of the packed arena.

    Pseudocode::

        placed = fixed obstacles
        for buffer in sorted(all but the small surviving buffers, key=size, descending):
            conflicts = [p for p in placed if lifetimes overlap]
            buffer.offset = lowest 512-aligned offset that misses every conflict
            placed.append(buffer)
        floor = end of that span
        for buffer in sorted(small surviving buffers, key=size, descending):
            # same, but offset >= floor; planning into an existing arena (`ceiling` = its size):
            # the highest free offset below the ceiling instead
            ...
        return max(offset + size)
    """
    placed: list[Buffer] = list(fixed)
    order = sorted(buffers, key=lambda buf: (-buf.size, buf.start, buf.index))

    def place(buf: Buffer, floor: int) -> None:
        conflicts = [other for other in placed if buf.overlaps(other)]
        conflicts.sort(key=lambda other: other.offset)
        cursor = floor
        for other in conflicts:
            if cursor + buf.size <= other.offset:
                break
            end = other.offset + other.size
            if end > cursor:
                cursor = end
        # Offsets stay on the allocator's alignment so a later carve never
        # produces a sub-512 sliver.
        if cursor % MIN_BLOCK:
            cursor = MIN_BLOCK * ((cursor + MIN_BLOCK - 1) // MIN_BLOCK)
        buf.offset = cursor
        placed.append(buf)

    def place_from_top(buf: Buffer, top_limit: int, floor: int) -> None:
        """Highest aligned offset below ``top_limit`` that misses every conflict; if that would dip
        under ``floor`` (the span of the other buffers), place upward from the floor instead."""
        conflicts = sorted((o for o in placed if buf.overlaps(o)), key=lambda o: -o.offset)
        off = ((top_limit - buf.size) // MIN_BLOCK) * MIN_BLOCK
        moved = True
        while moved and off >= floor:
            moved = False
            for o in conflicts:
                if off < o.offset + o.size and o.offset < off + buf.size:
                    off = ((o.offset - buf.size) // MIN_BLOCK) * MIN_BLOCK
                    moved = True
        if off < floor:
            place(buf, floor)
        else:
            buf.offset = off
            placed.append(buf)

    # Small buffers that outlive the capture (graph outputs) go above everything else: they stay
    # allocated after the capture, so wherever they sit they are an obstacle for the plan of every
    # later capture on this arena, and in the middle of the span they would split the free space a
    # later large buffer needs. On top they block nothing and cost their own size. Large survivors
    # take part in the normal greedy placement (stacking them on top would waste span).
    total = sum(buf.size for buf in buffers)
    limit = max(TOP_ZONE_MIN, total // TOP_ZONE_FRACTION)
    top = [buf for buf in order if buf.end is None and buf.size <= limit]
    main = [buf for buf in order if not (buf.end is None and buf.size <= limit)]
    for buf in main:
        place(buf, 0)
    floor = max((buf.offset + buf.size for buf in main), default=0)
    for buf in top:
        if ceiling is None:
            place(buf, floor)
        else:
            place_from_top(buf, ceiling, floor)
    if not buffers:
        return 0
    return max(buf.offset + buf.size for buf in buffers)


def plan_from_events(events: list[dict], fixed: list[Buffer] = (), ceiling: int | None = None) -> dict:
    """Plan the buffers of one trace. ``fixed`` are obstacles that already occupy parts of the
    arena (planning into an arena that earlier captures used) and ``ceiling`` is that arena's size;
    ``packed_bytes`` is the end of the span used by the new buffers only."""
    buffers = buffers_from_trace(events)
    packed = greedy_by_size(buffers, fixed, ceiling)
    # One cudaMalloc segment has to be large enough for the packed span and
    # has to match get_allocation_size, or the caching allocator rounds it up
    # on its own and the carve offsets no longer cover the segment.
    arena = arena_size(packed) if packed else 0
    ordered = sorted(buffers, key=lambda buf: buf.index)
    return {
        "buffers": ordered,
        "rounded_sizes": [buf.size for buf in ordered],
        "offsets": [buf.offset for buf in ordered],
        "packed_bytes": packed,
        "arena_bytes": arena,
        "peak_live_bytes": peak_live_bytes(buffers),
        "n_buffers": len(buffers),
    }


def verify_plan(buffers: list[Buffer], arena_bytes: int) -> dict:
    """Check the placement invariants. Raises ``AssertionError`` on the first violation.

    1. every buffer lies inside ``[0, arena_bytes)`` and is MIN_BLOCK aligned;
    2. two buffers whose lifetimes overlap are disjoint in address.

    Invariant 2 is what makes the packed capture equivalent to the recorded one: no
    buffer can observe a write to another buffer that is live at the same time.
    Cost is O(n * active) with a sweep over lifetime starts.
    """
    for buf in buffers:
        assert buf.offset % MIN_BLOCK == 0, f"buffer {buf.index}: offset {buf.offset} not aligned"
        assert buf.offset >= 0 and buf.offset + buf.size <= arena_bytes, (
            f"buffer {buf.index}: [{buf.offset}, {buf.offset + buf.size}) outside arena {arena_bytes}"
        )
    inf = 1 << 62
    order = sorted(buffers, key=lambda b: (b.start, b.index))
    active: list[Buffer] = []
    checked = 0
    for buf in order:
        active = [a for a in active if (a.end if a.end is not None else inf) > buf.start]
        for other in active:
            checked += 1
            assert buf.offset + buf.size <= other.offset or other.offset + other.size <= buf.offset, (
                f"buffers {buf.index} and {other.index} are live together but share addresses: "
                f"[{buf.offset}, {buf.offset + buf.size}) vs [{other.offset}, {other.offset + other.size})"
            )
        active.append(buf)
    return {"buffers": len(buffers), "pairs_checked": checked, "arena_bytes": arena_bytes}
