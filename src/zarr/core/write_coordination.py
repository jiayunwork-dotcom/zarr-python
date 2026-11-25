"""
Optional coordination of concurrent writes to the same physical store key.

Scope
-----
Coordination is opt-in (``ArrayConfig.write_synchronized``) and only covers array
*data* writes -- the bytes of one chunk, or of a whole shard when a sharding codec is
present -- performed by array handles that

* share the same ``Store`` *instance* (identity, not equality), and
* run on the same asyncio event loop.

Writes to different keys never wait for each other, and a write in flight never blocks
writes to other keys. Nothing here is shared across store instances, event loops,
threads or processes, and array metadata is never touched.

Design
------
``WriteCoordinator`` keeps a registry mapping (event loop, store instance) to a
``_StoreRegistry``; every ``_StoreRegistry`` holds a table of per-key records
(``_KeyRecord``). A writer calls :meth:`WriteCoordinator.acquire`, which synchronously
increments the key's *reference* count before anything is awaited, and enters the
returned async context manager around its read-modify-write critical section.

The per-key mutual exclusion is a hand-rolled queue of wakeup futures rather than an
``asyncio.Lock``. The reason is cancellation: ``asyncio.Lock.acquire`` can hand the lock
to a waiter in the same event-loop step in which that waiter is cancelled, which can
leave the lock permanently held. Here releasing wakes every waiter and each one
re-checks the owner synchronously, claiming the key at most once; a cancelled waiter
therefore can never strand it. The reference count is taken before the first ``await``
and returned on every exit path (including cancellation while waiting and cancellation
of the critical section), so a record is dropped from the table as soon as no task
references it, and the per-store registry is dropped once it holds no records. Writing
many distinct keys therefore does not grow the tables.

Which byte setters are physical?
--------------------------------
Only byte setters whose physical storage identity is known participate.
``StorePath`` exposes ``.store`` and ``.path``, which together name one physical key.
The sharding codec's internal byte setters address an in-memory shard builder and have
no such identity, so they are ignored; the enclosing shard write/delete is then the
unit of coordination. This is also why nested codecs never take a second, unrelated
lock (and cannot deadlock).

The bookkeeping objects are mutated only in synchronous code between ``await`` points,
so within one event loop the acquire/release sequence is atomic with respect to other
tasks.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.abc.store import Store

__all__ = [
    "WriteCoordinator",
    "get_write_coordinator",
    "physical_target",
]


@dataclass
class _KeyRecord:
    """Bookkeeping for one physical key of one store on one event loop."""

    refs: int = 0
    """Number of tasks that have entered :meth:`WriteCoordinator.acquire` and not left."""
    owner: object | None = None
    """Identity of the task currently inside the critical section, if any."""
    depth: int = 0
    """Re-entrancy depth of the owner (>1 if the same task nested an acquisition)."""
    waiters: list[asyncio.Future[None]] = field(default_factory=list)
    """Wakeup futures of tasks queueing for the critical section."""


def _current_owner() -> object:
    task = asyncio.current_task()
    # Without a task (a coroutine driven by hand) every acquisition gets a fresh owner
    # token, so two unrelated acquisitions are never mistaken for a nested one.
    return object() if task is None else task


class _StoreRegistry:
    """Coordination state for one (event loop, store instance) pair."""

    __slots__ = ("records",)

    def __init__(self) -> None:
        self.records: dict[str, _KeyRecord] = {}

    def get_or_create(self, key: str) -> _KeyRecord:
        record = self.records.get(key)
        if record is None:
            record = _KeyRecord()
            self.records[key] = record
        return record

    def maybe_discard(self, key: str, record: _KeyRecord) -> None:
        # A record with no references is neither held nor waited on; drop it so the
        # registry does not retain one entry for every key that was ever written.
        if record.refs == 0 and self.records.get(key) is record:
            del self.records[key]


class _KeyLock:
    """Async context manager guarding one physical key."""

    __slots__ = ("_acquired", "_coordinator", "_key", "_loop", "_record", "_reentrant", "_store")

    def __init__(
        self,
        coordinator: WriteCoordinator,
        loop: asyncio.AbstractEventLoop,
        store: Store,
        key: str,
        record: _KeyRecord,
    ) -> None:
        self._coordinator = coordinator
        self._loop = loop
        self._store = store
        self._key = key
        self._record = record
        self._acquired = False
        self._reentrant = False

    async def __aenter__(self) -> None:
        record = self._record
        owner = _current_owner()
        try:
            while record.owner is not None:
                if record.owner is owner:
                    # Same task nested an acquisition (nested codecs do not normally
                    # participate, but a custom codec might). Count it, do not re-lock.
                    record.depth += 1
                    self._reentrant = True
                    self._acquired = True
                    return
                fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                record.waiters.append(fut)
                try:
                    await fut
                finally:
                    with contextlib.suppress(ValueError):
                        record.waiters.remove(fut)
            record.owner = owner
            record.depth = 1
        except BaseException:
            # Interrupted before taking ownership. Give back the reference taken by
            # ``acquire`` here, since ``__aexit__`` does not run for a failed
            # ``__aenter__``.
            self._coordinator._release(self._loop, self._store, self._key, self._record)
            raise
        self._acquired = True

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            record = self._record
            if self._reentrant:
                record.depth -= 1
            elif self._acquired and record.owner is _current_owner():
                record.depth -= 1
                if record.depth == 0:
                    record.owner = None
                    # Wake every waiter: each one re-checks the owner synchronously, so
                    # exactly one claims the key and a cancelled waiter cannot strand it.
                    waiters, record.waiters = record.waiters, []
                    for fut in waiters:
                        if not fut.done():
                            fut.set_result(None)
        finally:
            self._acquired = False
            self._reentrant = False
            self._coordinator._release(self._loop, self._store, self._key, self._record)


class WriteCoordinator:
    """
    Registry of per-store, per-key write locks.

    All array handles of a process share one coordinator via
    :func:`get_write_coordinator`, which is what makes coordination effective between
    two handles opened on the same ``Store`` instance. The coordinator itself holds no
    per-array state; everything is keyed by storage identity.
    """

    __slots__ = ("_store_state",)

    def __init__(self) -> None:
        # Keyed by (loop id, store id); values keep strong references to both so that
        # neither id can be reused by a live object while the entry exists.
        self._store_state: dict[
            tuple[int, int], tuple[asyncio.AbstractEventLoop, Store, _StoreRegistry]
        ] = {}

    def _registry_for(self, loop: asyncio.AbstractEventLoop, store: Store) -> _StoreRegistry:
        state_key = (id(loop), id(store))
        entry = self._store_state.get(state_key)
        if entry is None or entry[0] is not loop or entry[1] is not store:
            registry = _StoreRegistry()
            self._store_state[state_key] = (loop, store, registry)
            return registry
        return entry[2]

    def acquire(self, store: Store, key: str) -> _KeyLock:
        """
        Reserve the coordination record for ``key`` of ``store``.

        Must be called from a running event loop. The reference is taken before any
        ``await``, so cancellation can never leave a dangling record.
        """
        loop = asyncio.get_running_loop()
        record = self._registry_for(loop, store).get_or_create(key)
        record.refs += 1
        return _KeyLock(self, loop, store, key, record)

    def _release(
        self,
        loop: asyncio.AbstractEventLoop,
        store: Store,
        key: str,
        record: _KeyRecord,
    ) -> None:
        record.refs -= 1
        state_key = (id(loop), id(store))
        entry = self._store_state.get(state_key)
        if entry is None or entry[0] is not loop or entry[1] is not store:
            return
        registry = entry[2]
        registry.maybe_discard(key, record)
        if not registry.records:
            # The store has no keys in flight: release the strong references too.
            del self._store_state[state_key]

    def n_tracked_keys(self, store: Store, *, loop: asyncio.AbstractEventLoop | None = None) -> int:
        """
        Number of keys currently tracked for ``store`` on ``loop`` (default: the running
        loop). Intended for diagnostics and tests.
        """
        if loop is None:
            loop = asyncio.get_running_loop()
        entry = self._store_state.get((id(loop), id(store)))
        if entry is None:
            return 0
        return len(entry[2].records)

    def n_tracked_stores(self) -> int:
        """Number of (loop, store) pairs with coordination state. Diagnostics/tests."""
        return len(self._store_state)


_GLOBAL_COORDINATOR = WriteCoordinator()


def get_write_coordinator() -> WriteCoordinator:
    """Return the process-wide write coordinator."""
    return _GLOBAL_COORDINATOR


def physical_target(byte_setter: object) -> tuple[Store, str] | None:
    """
    Return the physical storage identity of ``byte_setter``, or ``None``.

    ``StorePath`` (and anything else exposing a ``Store`` in ``.store`` and a key in
    ``.path``) names one physical key and participates in coordination. Byte setters
    internal to a codec -- e.g. the sharding codec's in-memory shard builder -- return
    ``None``; the enclosing store-level write is then the unit of coordination.
    """
    from zarr.abc.store import Store
    from zarr.storage._common import StorePath

    if isinstance(byte_setter, StorePath):
        return (byte_setter.store, byte_setter.path)
    store = getattr(byte_setter, "store", None)
    path = getattr(byte_setter, "path", None)
    if isinstance(store, Store) and isinstance(path, str):
        return (store, path)
    return None
