.. _user-guide-write-synchronization:

Coordinating concurrent writes
==============================

Zarr writes a chunk with a read-modify-write cycle: the existing block is read, the new
selection is merged into it, the result is encoded, and the bytes are stored back. If two
array handles write *different elements of the same chunk* at the same time, both may read
the old block and the second write-back can silently discard the first update.

As an example, two handles opened on one store write disjoint elements of one chunk::

   >>> import asyncio
   >>> import zarr
   >>> async def example():
   ...     store = zarr.storage.MemoryStore()
   ...     a = await zarr.api.asynchronous.create_array(
   ...         store, shape=(8,), chunks=(8,), dtype="int32", fill_value=0,
   ...         config={"write_synchronized": True},
   ...     )
   ...     await a.setitem(..., 1)
   ...     b = await zarr.AsyncArray.open(store, config={"write_synchronized": True})
   ...     await asyncio.gather(a.setitem((0,), 11), b.setitem((3,), 33))
   ...     return (await a.getitem(...)).tolist()
   >>> example_data = asyncio.run(example())
   >>> example_data
   [11, 1, 1, 33, 1, 1, 1, 1]

With coordination enabled both updates survive, regardless of how the two writes are
interleaved.

Enabling it
-----------

``write_synchronized`` is a *runtime* array setting, like ``write_empty_chunks``. It is
never written to the array metadata, so it applies to the handles you configure, not to the
data file:

.. code-block:: python

   # at creation
   arr = zarr.create_array(store, shape=(8,), chunks=(8,), dtype="int32",
                           config={"write_synchronized": True})

   # when opening, synchronously or asynchronously
   arr = zarr.open_array(store, config={"write_synchronized": True})
   arr = zarr.open(store, config={"write_synchronized": True})
   arr = zarr.Array.open(store, config={"write_synchronized": True})
   arr = await zarr.AsyncArray.open(store, config={"write_synchronized": True})

   # globally, for every array that does not override it
   zarr.config.set({"array.write_synchronized": True})

The setting follows the usual array-configuration resolution: an explicit ``config`` wins;
otherwise ``zarr.config["array.write_synchronized"]`` (default ``False``) is used. The
equivalent environment variable is ``ZARR_ARRAY__WRITE_SYNCHRONIZED=True``.

What is guaranteed
------------------

For handles that share the same ``Store`` **instance** and run on the same event loop, and
that all enable the mode:

* Modifying the same physical store key is equivalent to *some* serial order of the
  writes. A partial update, a whole-block overwrite and a fill-value write that deletes the
  block all count as modifications.
* With sharding, the physical unit is the whole shard: two writes that touch different
  inner chunks of the same shard are serialized, and both take effect.
* Different keys make progress independently. A chunk whose store operation is slow does
  not block writes to other chunks or other shards.
* A writer that is cancelled while waiting, or whose encoding or store write fails, does
  not block later writers. Bytes already written are not rolled back.

What is not guaranteed:

* Writes spanning several keys are not atomic as a group.
* Handles that created their own ``Store`` object (even over the same underlying data), or
  that run on different event loops, threads or processes, are not coordinated.
* Array metadata changes, ``resize``/``append`` and distributed coordination are out of
  scope.

How it works, and what it costs
-------------------------------

The coordination state lives in a process-wide registry (``zarr.core.write_coordination``)
keyed by event loop and by ``Store`` identity, with one lock per physical key. The codec
pipeline takes that lock around the whole read-merge-encode-store sequence of a chunk, or
of a shard when a sharding codec is present. Nested codecs address an in-memory shard
builder rather than the store, so they do not participate and cannot deadlock.

Lock bookkeeping is reference counted and taken before the first ``await``; a record is
removed as soon as no task holds or waits for it, and the per-store registry disappears
once it has no keys. Writing many different keys therefore does not grow the registry.
Inside a process the only extra cost is a Python-level acquire per written key and the loss
of parallelism between writers of the *same* key; writes to different keys stay
concurrent. The registry holds a strong reference to a store only while that store has
writes in flight.
