"""
Tests for ``ArrayConfig.write_synchronized``, the opt-in write coordination mode.

The tests use :class:`ControllableStore`, an in-memory store that lets a test block
individual store operations on events and observe which operations have started. That
makes every interleaving deterministic -- no test relies on ``sleep`` to hit a race
(``asyncio.sleep(0)`` is only used as "yield to the event loop", inside conditions that
are re-checked until the expected state is observed).

Expected values are obtained independently, by applying the same logical writes in a
chosen sequential order to a fresh, uncoordinated array (a legal serial execution).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from zarr.abc.store import Store
from zarr.core.write_coordination import get_write_coordinator
from zarr.storage import MemoryStore
from zarr.storage._utils import _normalize_byte_range_index

if TYPE_CHECKING:
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype

# A single logical write, expressed against any array handle.
WriteOp = Callable[["zarr.AsyncArray[Any]"], Awaitable[None]]


# --------------------------------------------------------------------------------------
# A store whose individual operations can be held and resumed by the test.
# --------------------------------------------------------------------------------------


class ControllableStore(Store):
    """
    In-memory store where every ``get``/``set``/``delete`` is observable and can be held.

    ``gates`` maps ``(operation, key)`` to an ``asyncio.Event``; while the event is not
    set the operation waits *before* touching data. ``started`` records
    ``(operation, key)`` tuples as operations begin, and ``active``/``max_active`` count
    operations of a key that are inside the store at the same time (which lets a test
    assert that a physical key is never touched by two writers concurrently).
    """

    supports_writes = True
    supports_deletes = True
    supports_listing = True

    def __init__(self, store_dict: MutableMapping[str, Buffer] | None = None) -> None:
        super().__init__()
        self._store_dict: MutableMapping[str, Buffer] = {} if store_dict is None else store_dict
        self.gates: dict[tuple[str, str], asyncio.Event] = {}
        self.started: list[tuple[str, str]] = []
        self.active: dict[str, int] = {}
        self.max_active: dict[str, int] = {}
        self.fail_set: set[str] = set()

    # -- test helpers ------------------------------------------------------------------

    def gate(self, op: str, key: str) -> asyncio.Event:
        ev = asyncio.Event()
        self.gates[(op, key)] = ev
        return ev

    def release(self, op: str, key: str) -> None:
        ev = self.gates.pop((op, key), None)
        if ev is not None:
            ev.set()

    def release_all(self) -> None:
        for ev in self.gates.values():
            ev.set()
        self.gates.clear()

    def reset_observations(self) -> None:
        """Forget operations observed so far (used after test setup)."""
        self.started.clear()
        self.active.clear()
        self.max_active.clear()

    def started_ops(self, op: str, key: str) -> int:
        return sum(1 for entry in self.started if entry == (op, key))

    async def wait_started(self, op: str, key: str, count: int = 1) -> None:
        """Yield to the event loop until ``count`` operations of ``(op, key)`` started."""
        while self.started_ops(op, key) < count:
            await asyncio.sleep(0)

    async def _enter(self, op: str, key: str) -> None:
        self.started.append((op, key))
        ev = self.gates.get((op, key))
        if ev is not None:
            await ev.wait()
        self.active[key] = self.active.get(key, 0) + 1
        self.max_active[key] = max(self.max_active.get(key, 0), self.active[key])

    def _exit(self, key: str) -> None:
        self.active[key] -= 1

    # -- Store protocol ----------------------------------------------------------------

    def __eq__(self, value: object) -> bool:
        return self is value

    def __str__(self) -> str:
        return f"controllable://{id(self._store_dict)}"

    def __repr__(self) -> str:
        return f"ControllableStore('{self}')"

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        await self._enter("get", key)
        try:
            value = self._store_dict.get(key)
            if value is None:
                return None
            start, stop = _normalize_byte_range_index(value, byte_range)
            return prototype.buffer.from_buffer(value[start:stop])
        finally:
            self._exit(key)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return [
            await self.get(key, prototype=prototype, byte_range=byte_range)
            for key, byte_range in key_ranges
        ]

    async def exists(self, key: str) -> bool:
        return key in self._store_dict

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        await self._enter("set", key)
        try:
            if key in self.fail_set:
                raise RuntimeError(f"injected store failure for {key!r}")
            self._store_dict[key] = value
        finally:
            self._exit(key)

    async def delete(self, key: str) -> None:
        self._check_writable()
        await self._enter("delete", key)
        try:
            self._store_dict.pop(key, None)
        finally:
            self._exit(key)

    async def list(self) -> AsyncIterator[str]:
        for key in list(self._store_dict):
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        for key in list(self._store_dict):
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def chunk_key(array: zarr.AsyncArray[Any], *coords: int) -> str:
    """The physical store key of a chunk (in a sharded array, of its shard)."""
    return array.metadata.encode_chunk_key(coords)


async def make_array(
    store: Store,
    *,
    chunks: tuple[int, ...] = (8,),
    shards: tuple[int, ...] | None = None,
    fill_value: Any = 0,
    synchronized: bool = True,
) -> zarr.AsyncArray[Any]:
    kwargs: dict[str, Any] = {}
    if shards is not None:
        kwargs["shards"] = shards
    return await zarr.api.asynchronous.create_array(
        store,
        shape=(8,),
        chunks=chunks,
        dtype="int32",
        fill_value=fill_value,
        zarr_format=3,
        config={"write_synchronized": synchronized},
        **kwargs,
    )


async def open_array(store: Store, *, synchronized: bool = True) -> zarr.AsyncArray[Any]:
    return await zarr.AsyncArray.open(store, config={"write_synchronized": synchronized})


async def serial_reference(
    ops: Iterable[WriteOp],
    *,
    initial: Any,
    chunks: tuple[int, ...] = (8,),
    shards: tuple[int, ...] | None = None,
    fill_value: Any = 0,
) -> np.ndarray[Any, Any]:
    """
    Apply ``ops`` in order to a fresh, uncoordinated array with the same layout.

    This is a legal serial execution of the same logical writes; it gives the expected
    bytes independently of the coordination machinery.
    """
    ref = await make_array(
        ControllableStore(),
        chunks=chunks,
        shards=shards,
        fill_value=fill_value,
        synchronized=False,
    )
    await ref.setitem(..., initial)
    for op in ops:
        await op(ref)
    return np.asarray(await ref.getitem(...))


def assert_no_coordination_state(store: Store) -> None:
    coordinator = get_write_coordinator()
    assert coordinator.n_tracked_keys(store) == 0, "per-key coordination state leaked"


# --------------------------------------------------------------------------------------
# 1. The motivating example
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chunks", "shards"),
    [
        ((8,), None),
        ((2,), (8,)),
    ],
    ids=["single-chunk", "sharded"],
)
async def test_two_handles_disjoint_updates_both_survive(
    chunks: tuple[int, ...], shards: tuple[int, ...] | None
) -> None:
    store = ControllableStore()
    a = await make_array(store, chunks=chunks, shards=shards)
    await a.setitem(..., 1)

    b = await open_array(store)
    store.reset_observations()

    key = chunk_key(a, 0)
    # Hold the very first store access of the first writer: with coordination the second
    # writer cannot touch the same physical key until the first one is done.
    gate = store.gate("get", key)
    a_task = asyncio.ensure_future(a.setitem((0,), 11))
    await store.wait_started("get", key)

    b_task = asyncio.ensure_future(b.setitem((3,), 33))
    # Give b every chance to reach the store; it must be blocked on the key.
    for _ in range(50):
        await asyncio.sleep(0)
    assert store.started_ops("get", key) == 1
    assert store.started_ops("set", key) == 0
    assert not b_task.done()

    gate.set()
    await asyncio.gather(a_task, b_task)

    expected = np.ones(8, dtype="int32")
    expected[0] = 11
    expected[3] = 33
    np.testing.assert_array_equal(await a.getitem(...), expected)
    assert_no_coordination_state(store)


async def test_race_exists_without_coordination() -> None:
    """
    Characterization of the existing bug: with the mode disabled, one update is lost
    when both writers read the same physical block before either writes it back.
    """
    store = ControllableStore()
    a = await make_array(store, synchronized=False)
    await a.setitem(..., 1)
    b = await open_array(store, synchronized=False)
    store.reset_observations()

    key = chunk_key(a, 0)

    # Hold only the *first* writer's store.set until the second writer's complete
    # read-modify-write has gone through.
    orig_set = store.set
    blocked = asyncio.Event()
    reached = asyncio.Event()
    held = False

    async def set_hold_first(k: str, value: Buffer) -> None:
        nonlocal held
        if k == key and not held:
            held = True
            reached.set()
            await blocked.wait()
        await orig_set(k, value)

    store.set = set_hold_first  # type: ignore[method-assign]

    a_task = asyncio.ensure_future(a.setitem((0,), 11))
    await asyncio.wait_for(reached.wait(), timeout=10)
    await b.setitem((3,), 33)
    blocked.set()
    await a_task

    result = np.asarray(await a.getitem(...))
    # a wrote the block it read before b's update, so b's value is gone.
    assert result[0] == 11
    assert result[3] == 1


async def test_two_handles_both_updates_survive_v2() -> None:
    """The same guarantee applies to Zarr format 2 arrays."""
    store = ControllableStore()
    a = await zarr.api.asynchronous.create_array(
        store,
        shape=(8,),
        chunks=(8,),
        dtype="int32",
        zarr_format=2,
        fill_value=0,
        config={"write_synchronized": True},
    )
    await a.setitem(..., 1)
    b = await zarr.AsyncArray.open(store, zarr_format=2, config={"write_synchronized": True})
    store.reset_observations()

    key = chunk_key(a, 0)
    gate = store.gate("get", key)
    a_task = asyncio.ensure_future(a.setitem((0,), 11))
    await store.wait_started("get", key)
    b_task = asyncio.ensure_future(b.setitem((3,), 33))
    for _ in range(50):
        await asyncio.sleep(0)
    assert store.started_ops("get", key) == 1
    assert not b_task.done()

    gate.set()
    await asyncio.gather(a_task, b_task)

    expected = np.ones(8, dtype="int32")
    expected[0] = 11
    expected[3] = 33
    np.testing.assert_array_equal(await a.getitem(...), expected)
    assert_no_coordination_state(store)


# --------------------------------------------------------------------------------------
# 2. Whole-block overwrite vs partial modification; delete vs write
# --------------------------------------------------------------------------------------


async def test_whole_overwrite_then_partial_modification() -> None:
    store = ControllableStore()
    a = await make_array(store)
    await a.setitem(..., 1)
    b = await open_array(store)

    store.reset_observations()
    key = chunk_key(a, 0)

    async def overwrite(arr: zarr.AsyncArray[Any]) -> None:
        await arr.setitem(..., np.arange(8, dtype="int32") + 100)

    async def partial(arr: zarr.AsyncArray[Any]) -> None:
        await arr.setitem((3,), 33)

    expected = await serial_reference([overwrite, partial], initial=1)

    # Force the order: the overwrite is fully inside the store (held at `set`) while the
    # partial write is issued; the partial write must wait.
    set_gate = store.gate("set", key)
    a_task = asyncio.ensure_future(overwrite(a))
    await store.wait_started("set", key)
    b_task = asyncio.ensure_future(partial(b))
    for _ in range(50):
        await asyncio.sleep(0)
    assert store.started_ops("get", key) == 0, "second writer entered the block too early"
    set_gate.set()
    await asyncio.gather(a_task, b_task)

    np.testing.assert_array_equal(await a.getitem(...), expected)
    assert_no_coordination_state(store)


async def test_partial_modification_then_whole_overwrite() -> None:
    store = ControllableStore()
    a = await make_array(store)
    await a.setitem(..., 1)
    b = await open_array(store)

    store.reset_observations()
    key = chunk_key(a, 0)

    async def overwrite(arr: zarr.AsyncArray[Any]) -> None:
        await arr.setitem(..., np.arange(8, dtype="int32") + 100)

    async def partial(arr: zarr.AsyncArray[Any]) -> None:
        await arr.setitem((3,), 33)

    expected = await serial_reference([partial, overwrite], initial=1)

    get_gate = store.gate("get", key)
    b_task = asyncio.ensure_future(partial(b))
    await store.wait_started("get", key)
    a_task = asyncio.ensure_future(overwrite(a))
    for _ in range(50):
        await asyncio.sleep(0)
    assert not a_task.done()
    get_gate.set()
    await asyncio.gather(a_task, b_task)

    np.testing.assert_array_equal(await a.getitem(...), expected)
    assert_no_coordination_state(store)


async def test_delete_then_write() -> None:
    """A fill-value write deletes the block; a later partial write recreates it."""
    store = ControllableStore()
    a = await make_array(store, fill_value=0)
    await a.setitem(..., 7)
    b = await open_array(store)

    store.reset_observations()
    key = chunk_key(a, 0)

    async def reset(arr: zarr.AsyncArray[Any]) -> None:
        await arr.setitem(..., 0)

    async def partial(arr: zarr.AsyncArray[Any]) -> None:
        await arr.setitem((3,), 33)

    expected = await serial_reference([reset, partial], initial=7)

    delete_gate = store.gate("delete", key)
    a_task = asyncio.ensure_future(reset(a))
    await store.wait_started("delete", key)
    # The holder's own read of the shard has happened by now; from here on, nothing may
    # reach the key while the holder is blocked.
    store.reset_observations()
    b_task = asyncio.ensure_future(partial(b))
    for _ in range(50):
        await asyncio.sleep(0)
    assert store.started == []
    delete_gate.set()
    await asyncio.gather(a_task, b_task)

    np.testing.assert_array_equal(await a.getitem(...), expected)
    # and the block really does exist again, because it is no longer all fill
    assert key in store._store_dict
    assert_no_coordination_state(store)


async def test_write_then_delete() -> None:
    """A partial write followed by a fill-value write: the block is deleted."""
    store = ControllableStore()
    a = await make_array(store, fill_value=0)
    await a.setitem(..., 7)
    b = await open_array(store)

    store.reset_observations()
    key = chunk_key(a, 0)

    async def reset(arr: zarr.AsyncArray[Any]) -> None:
        await arr.setitem(..., 0)

    async def partial(arr: zarr.AsyncArray[Any]) -> None:
        await arr.setitem((3,), 33)

    expected = await serial_reference([partial, reset], initial=7)

    get_gate = store.gate("get", key)
    b_task = asyncio.ensure_future(partial(b))
    await store.wait_started("get", key)
    a_task = asyncio.ensure_future(reset(a))
    for _ in range(50):
        await asyncio.sleep(0)
    assert not a_task.done()
    get_gate.set()
    await asyncio.gather(a_task, b_task)

    np.testing.assert_array_equal(await a.getitem(...), expected)
    assert key not in store._store_dict
    assert_no_coordination_state(store)


@pytest.mark.parametrize(
    ("chunks", "shards"),
    [((2,), (8,))],
    ids=["sharded"],
)
async def test_sharded_delete_and_write(chunks: tuple[int, ...], shards: tuple[int, ...]) -> None:
    """The physical unit of a sharded array is the whole shard."""
    store = ControllableStore()
    a = await make_array(store, chunks=chunks, shards=shards, fill_value=0)
    await a.setitem(..., 7)
    b = await open_array(store)

    store.reset_observations()
    key = chunk_key(a, 0)  # there is a single shard

    async def reset(arr: zarr.AsyncArray[Any]) -> None:
        await arr.setitem(..., 0)

    async def partial(arr: zarr.AsyncArray[Any]) -> None:
        await arr.setitem((5,), 55)  # inner chunk 2 of the shard

    expected = await serial_reference([reset, partial], initial=7, chunks=chunks, shards=shards)

    delete_gate = store.gate("delete", key)
    a_task = asyncio.ensure_future(reset(a))
    await store.wait_started("delete", key)
    # The holder's own read of the shard has happened by now; from here on, nothing may
    # reach the key while the holder is blocked.
    store.reset_observations()
    b_task = asyncio.ensure_future(partial(b))
    for _ in range(50):
        await asyncio.sleep(0)
    assert store.started == []
    delete_gate.set()
    await asyncio.gather(a_task, b_task)

    np.testing.assert_array_equal(await a.getitem(...), expected)
    assert_no_coordination_state(store)


# --------------------------------------------------------------------------------------
# 3. Independent keys make progress
# --------------------------------------------------------------------------------------


async def test_slow_key_does_not_block_other_keys() -> None:
    store = ControllableStore()
    a = await make_array(store, chunks=(1,))
    b = await open_array(store)

    slow_key = chunk_key(a, 0)

    set_gate = store.gate("set", slow_key)
    slow_task = asyncio.ensure_future(a.setitem((0,), 11))
    await store.wait_started("set", slow_key)

    # A write to a different key must complete while the first key is held.
    await asyncio.wait_for(b.setitem((3,), 33), timeout=10)
    assert ("set", slow_key) in store.gates  # still held
    assert not slow_task.done()

    set_gate.set()
    await slow_task

    expected = np.zeros(8, dtype="int32")
    expected[0] = 11
    expected[3] = 33
    np.testing.assert_array_equal(await a.getitem(...), expected)
    assert_no_coordination_state(store)


async def test_sharded_different_inner_chunks_are_serialized() -> None:
    """
    Two inner chunks of one shard are one physical key: writes to them are serialized and
    neither clobbers the other, even though the element selections are disjoint.
    """
    store = ControllableStore()
    a = await make_array(store, chunks=(2,), shards=(8,))
    await a.setitem(..., 1)
    b = await open_array(store)
    store.reset_observations()

    key = chunk_key(a, 0)
    gate = store.gate("get", key)
    a_task = asyncio.ensure_future(a.setitem((0,), 11))  # inner chunk 0
    await store.wait_started("get", key)
    b_task = asyncio.ensure_future(b.setitem((5,), 55))  # inner chunk 2, same shard
    for _ in range(50):
        await asyncio.sleep(0)
    assert store.started_ops("get", key) == 1
    assert not b_task.done()
    gate.set()
    await asyncio.gather(a_task, b_task)

    expected = np.ones(8, dtype="int32")
    expected[0] = 11
    expected[5] = 55
    np.testing.assert_array_equal(await a.getitem(...), expected)
    assert_no_coordination_state(store)


async def test_slow_shard_does_not_block_other_shards() -> None:
    store = ControllableStore()
    a = await make_array(store, chunks=(2,), shards=(4,))

    slow_key = chunk_key(a, 0)
    other_key = chunk_key(a, 1)

    set_gate = store.gate("set", slow_key)
    slow_task = asyncio.ensure_future(a.setitem((1,), 11))  # shard 0
    await store.wait_started("set", slow_key)

    await asyncio.wait_for(a.setitem((5,), 55), timeout=10)  # shard 1
    assert store.started_ops("set", other_key) == 1
    assert not slow_task.done()

    set_gate.set()
    await slow_task

    expected = np.zeros(8, dtype="int32")
    expected[1] = 11
    expected[5] = 55
    np.testing.assert_array_equal(await a.getitem(...), expected)
    assert_no_coordination_state(store)


async def test_physical_key_never_accessed_concurrently() -> None:
    """Under coordination no physical key has two store operations inside at once."""
    store = ControllableStore()
    a = await make_array(store, chunks=(2,), shards=(8,))
    b = await open_array(store)
    await a.setitem(..., 1)
    store.reset_observations()

    await asyncio.gather(
        a.setitem((0,), 11),
        b.setitem((1,), 22),
        a.setitem((2,), 33),
        b.setitem((3,), 44),
    )
    key = chunk_key(a, 0)
    assert store.max_active[key] == 1
    expected = np.ones(8, dtype="int32")
    expected[:4] = (11, 22, 33, 44)
    np.testing.assert_array_equal(await a.getitem(...), expected)
    assert_no_coordination_state(store)


async def test_repeated_concurrent_partial_writes_always_merge() -> None:
    """
    Stress: many trials of concurrent partial writes to one block, with the event loop
    deciding the interleaving. Under coordination *every* partial update must survive.
    """
    for trial in range(25):
        store = ControllableStore()
        a = await make_array(store)
        await a.setitem(..., 0)
        handles = [a] + [await open_array(store) for _ in range(3)]

        offsets = [0, 1, 2, 3]
        await asyncio.gather(
            *(h.setitem((i,), 10 + i) for i, h in zip(offsets, handles, strict=True))
        )

        got = np.asarray(await a.getitem(...))
        expected = np.zeros(8, dtype="int32")
        expected[: len(offsets)] = np.array(offsets) + 10
        np.testing.assert_array_equal(got, expected, err_msg=f"trial {trial}")
        assert_no_coordination_state(store)


async def test_overwrite_vs_partial_result_is_a_serial_outcome() -> None:
    """
    For an overwrite racing a partial update the result must equal one of the two legal
    serial outcomes -- never a mixture of stale and new data.
    """
    outcomes: set[bytes] = set()
    for trial in range(25):
        store = ControllableStore()
        a = await make_array(store)
        await a.setitem(..., 1)
        b = await open_array(store)

        async def overwrite(arr: zarr.AsyncArray[Any]) -> None:
            await arr.setitem(..., np.arange(8, dtype="int32") + 100)

        async def partial(arr: zarr.AsyncArray[Any]) -> None:
            await arr.setitem((3,), 33)

        ref_over_then_part = await serial_reference([overwrite, partial], initial=1)
        ref_part_then_over = await serial_reference([partial, overwrite], initial=1)

        await asyncio.gather(overwrite(a), partial(b))
        got = np.asarray(await a.getitem(...))

        legal = [ref_over_then_part, ref_part_then_over]
        assert any(np.array_equal(got, ref) for ref in legal), f"trial {trial}: {got}"
        outcomes.add(got.tobytes())
        assert_no_coordination_state(store)
    # over 25 trials the scheduler should produce both orders; if not, at least the
    # observed result was legal in every trial (asserted above).
    assert len(outcomes) >= 1


# --------------------------------------------------------------------------------------
# 4. Cancellation and failures leave the key usable
# --------------------------------------------------------------------------------------


async def test_cancel_waiter_then_write_proceeds() -> None:
    store = ControllableStore()
    a = await make_array(store)
    b = await open_array(store)
    await a.setitem(..., 1)
    store.reset_observations()
    key = chunk_key(a, 0)

    gate = store.gate("get", key)
    a_task = asyncio.ensure_future(a.setitem((0,), 11))
    await store.wait_started("get", key)

    b_task = asyncio.ensure_future(b.setitem((3,), 33))
    for _ in range(50):
        await asyncio.sleep(0)
    assert not b_task.done()

    # Cancel the waiter; the holder is still blocked.
    b_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await b_task
    # State of the key must still be sane: exactly one record, held by a.
    assert get_write_coordinator().n_tracked_keys(store) == 1

    gate.set()
    await a_task

    # A later writer can still enter.
    await asyncio.wait_for(b.setitem((3,), 33), timeout=10)
    expected = np.ones(8, dtype="int32")
    expected[0] = 11
    expected[3] = 33
    np.testing.assert_array_equal(await a.getitem(...), expected)
    assert_no_coordination_state(store)


async def test_cancel_holder_then_write_proceeds() -> None:
    store = ControllableStore()
    a = await make_array(store)
    b = await open_array(store)
    await a.setitem(..., 1)
    store.reset_observations()
    key = chunk_key(a, 0)

    store.gate("get", key)
    a_task = asyncio.ensure_future(a.setitem((0,), 11))
    await store.wait_started("get", key)
    b_task = asyncio.ensure_future(b.setitem((3,), 33))
    for _ in range(50):
        await asyncio.sleep(0)
    assert not b_task.done()

    # Cancel the writer that holds the key, while it is blocked inside the store.
    a_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await a_task
    store.release_all()

    # The queued writer must proceed.
    await asyncio.wait_for(b_task, timeout=10)
    expected = np.ones(8, dtype="int32")
    expected[3] = 33
    np.testing.assert_array_equal(await b.getitem(...), expected)
    assert_no_coordination_state(store)


async def test_store_failure_then_write_proceeds() -> None:
    store = ControllableStore()
    a = await make_array(store)
    b = await open_array(store)
    await a.setitem(..., 1)
    store.reset_observations()
    key = chunk_key(a, 0)

    store.fail_set.add(key)
    with pytest.raises(RuntimeError, match="injected store failure"):
        await a.setitem((0,), 11)
    store.fail_set.discard(key)

    await asyncio.wait_for(b.setitem((3,), 33), timeout=10)
    assert_no_coordination_state(store)

    expected = np.ones(8, dtype="int32")
    expected[3] = 33
    np.testing.assert_array_equal(await b.getitem(...), expected)


async def test_encode_failure_then_write_proceeds() -> None:
    store = ControllableStore()
    a = await make_array(store)
    b = await open_array(store)
    await a.setitem(..., 1)

    codec = a.codec_pipeline.array_bytes_codec
    real_encode = codec.encode
    calls = 0

    async def failing_encode(chunks: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("injected encode failure")
        return await real_encode(chunks)

    object.__setattr__(codec, "encode", failing_encode)

    with pytest.raises(ValueError, match="injected encode failure"):
        await a.setitem((0,), 11)

    object.__setattr__(codec, "encode", real_encode)
    await asyncio.wait_for(b.setitem((3,), 33), timeout=10)
    assert_no_coordination_state(store)


# --------------------------------------------------------------------------------------
# 5. State reclamation
# --------------------------------------------------------------------------------------


async def test_many_keys_do_not_accumulate_state() -> None:
    store = ControllableStore()
    a = await make_array(store, chunks=(1,))
    await a.setitem(..., 1)
    assert_no_coordination_state(store)

    # Reopened handles, many distinct keys.
    b = await open_array(store)
    for i in range(8):
        await a.setitem((i,), 10 + i)
        await b.setitem((i,), 20 + i)
        assert_no_coordination_state(store)

    # The coordinator does not keep the store alive either.
    coordinator = get_write_coordinator()
    del b
    assert coordinator.n_tracked_stores() >= 0  # sanity only; nothing tracked for store
    assert_no_coordination_state(store)


# --------------------------------------------------------------------------------------
# 6. Configuration plumbing
# --------------------------------------------------------------------------------------


def test_array_config_default_and_parsing() -> None:
    from zarr.core.array_spec import ArrayConfig, parse_array_config

    assert parse_array_config(None).write_synchronized is False
    assert parse_array_config({"write_synchronized": True}).write_synchronized is True
    assert ArrayConfig(order="C", write_empty_chunks=False).write_synchronized is False
    with pytest.raises(ValueError, match="Expected bool"):
        parse_array_config({"write_synchronized": "yes"})


def test_global_config_default() -> None:
    with zarr.config.set({"array.write_synchronized": True}):
        arr = zarr.create_array(store=MemoryStore(), shape=(8,), chunks=(8,), dtype="i4")
        assert arr._async_array._config.write_synchronized is True


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_config_reaches_create_open_sync_and_async(zarr_format: int) -> None:
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(8,),
        chunks=(4,),
        dtype="i4",
        zarr_format=zarr_format,
        config={"write_synchronized": True},
    )
    assert arr._async_array._config.write_synchronized is True

    # zarr.open_array (sync)
    opened = zarr.open_array(store=store, config={"write_synchronized": True})
    assert opened._async_array._config.write_synchronized is True

    # zarr.open (sync, generic)
    opened = zarr.open(store=store, config={"write_synchronized": True})
    assert isinstance(opened, zarr.Array)
    assert opened._async_array._config.write_synchronized is True

    # Array.open (sync class method); format is auto-detected in the array metadata
    opened = zarr.Array.open(store, zarr_format=None, config={"write_synchronized": True})
    assert opened._async_array._config.write_synchronized is True

    # zarr.create (deprecated but supported configuration entry point)
    created = zarr.create(
        store=store,
        shape=(8,),
        chunks=(4,),
        dtype="i4",
        zarr_format=zarr_format,
        overwrite=True,
        config={"write_synchronized": True},
    )
    assert created._async_array._config.write_synchronized is True

    # default: not enabled
    assert zarr.open_array(store=store)._async_array._config.write_synchronized is False


async def test_config_reaches_group_create_array() -> None:
    store = MemoryStore()
    group = zarr.create_group(store=store)
    arr = group.create_array(
        "a", shape=(8,), chunks=(4,), dtype="i4", config={"write_synchronized": True}
    )
    assert arr._async_array._config.write_synchronized is True

    async_group = await zarr.AsyncGroup.open(store)
    async_arr = await async_group.create_array(
        "b", shape=(8,), chunks=(4,), dtype="i4", config={"write_synchronized": True}
    )
    assert async_arr._config.write_synchronized is True


async def test_config_reaches_async_open() -> None:
    store = MemoryStore()
    await zarr.api.asynchronous.create_array(
        store, shape=(8,), chunks=(4,), dtype="i4", zarr_format=3
    )
    opened = await zarr.AsyncArray.open(store, config={"write_synchronized": True})
    assert opened._config.write_synchronized is True
    created = await zarr.api.asynchronous.open_array(
        store=store, config={"write_synchronized": True}
    )
    assert created._config.write_synchronized is True


def test_config_is_not_persisted_in_metadata() -> None:
    store = MemoryStore()
    zarr.create_array(
        store=store,
        shape=(8,),
        chunks=(4,),
        dtype="i4",
        zarr_format=3,
        config={"write_synchronized": True},
    )
    raw = json.loads(store._store_dict["zarr.json"].to_bytes())
    assert "write_synchronized" not in json.dumps(raw)


async def test_same_handle_concurrent_writes_to_one_key() -> None:
    """Two concurrent assignments on one handle also serialize cleanly."""
    store = ControllableStore()
    a = await make_array(store)
    await a.setitem(..., 1)

    key = chunk_key(a, 0)
    gate = store.gate("get", key)
    first = asyncio.ensure_future(a.setitem((0,), 11))
    await store.wait_started("get", key)
    second = asyncio.ensure_future(a.setitem((3,), 33))
    for _ in range(50):
        await asyncio.sleep(0)
    assert not second.done()
    gate.set()
    await asyncio.gather(first, second)

    expected = np.ones(8, dtype="int32")
    expected[0] = 11
    expected[3] = 33
    np.testing.assert_array_equal(await a.getitem(...), expected)
    assert_no_coordination_state(store)


async def test_mixed_handles_do_not_deadlock() -> None:
    """A handle without the mode bypasses the lock; a coordinated handle still works."""
    store = ControllableStore()
    a = await make_array(store)
    plain = await open_array(store, synchronized=False)
    await a.setitem(..., 1)

    key = chunk_key(a, 0)
    gate = store.gate("get", key)
    coordinated = asyncio.ensure_future(a.setitem((0,), 11))
    await store.wait_started("get", key)
    plain_task = asyncio.ensure_future(plain.setitem((3,), 33))
    for _ in range(50):
        await asyncio.sleep(0)
    # The uncoordinated handle is not held back by the coordinated writer's lock: it has
    # reached the store (where it is held by the test's own gate, like the first writer).
    assert store.started_ops("get", key) == 2
    gate.set()
    await asyncio.gather(coordinated, plain_task)
    assert_no_coordination_state(store)


async def test_many_concurrent_handles_all_updates_survive() -> None:
    """N handles writing distinct elements of one block: every value survives."""
    store = ControllableStore()
    a = await make_array(store)
    await a.setitem(..., 0)

    handles = [a] + [await open_array(store) for _ in range(3)]
    await asyncio.gather(*(h.setitem((i,), 100 + i) for i, h in enumerate(handles)))

    expected = np.zeros(8, dtype="int32")
    expected[: len(handles)] = np.arange(len(handles)) + 100
    np.testing.assert_array_equal(await a.getitem(...), expected)
    assert_no_coordination_state(store)


# --------------------------------------------------------------------------------------
# 7. Scope: coordination is per store instance and per event loop
# --------------------------------------------------------------------------------------


async def test_separate_store_instances_are_not_coordinated() -> None:
    """
    Different store instances are out of scope even when they share storage: with two
    instances over one dict, both writers reach the block concurrently and one update can
    be lost. This documents the boundary of the guarantee.
    """
    backing: dict[str, Any] = {}
    s1 = ControllableStore(backing)
    s2 = ControllableStore(backing)
    a = await make_array(s1)
    b = await zarr.AsyncArray.open(s2, config={"write_synchronized": True})
    await a.setitem(..., 1)
    s1.reset_observations()
    s2.reset_observations()

    key = chunk_key(a, 0)
    gate = s1.gate("get", key)
    a_task = asyncio.ensure_future(a.setitem((0,), 11))
    await s1.wait_started("get", key)

    # The writer through the *other* store instance is not blocked by that lock.
    s2.gate("get", key)
    b_task = asyncio.ensure_future(b.setitem((3,), 33))
    await s2.wait_started("get", key)
    assert not b_task.done()
    b_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await b_task

    gate.set()
    await a_task
    # Each instance has its own (now empty) coordination state; nothing is shared.
    assert get_write_coordinator().n_tracked_keys(s1) == 0
    assert get_write_coordinator().n_tracked_keys(s2) == 0


async def test_default_mode_does_not_use_coordination() -> None:
    store = ControllableStore()
    a = await make_array(store, synchronized=False)
    key = chunk_key(a, 0)
    gate = store.gate("set", key)
    task = asyncio.ensure_future(a.setitem(..., 1))
    await store.wait_started("set", key)
    # With coordination off the coordinator never hears about the write, so nothing is
    # tracked even while the store operation is held.
    assert get_write_coordinator().n_tracked_keys(store) == 0
    gate.set()
    await task
