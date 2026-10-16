# Architecture

```
dtrain/
  autograd.py   reverse-mode autograd on NumPy; grad-ready hooks on leaves
  nn.py         Linear, LayerNorm, causal self-attention, GPT
  optim.py      AdamW on flat arrays (shared by DDP and ZeRO); global-norm clipping
  comm.py       TCP rendezvous store, full-mesh process group, ring/direct collectives, async work queue
  ddp.py        distributed data parallel: param broadcast, gradient buckets, overlap, no_sync
  zero.py       ZeRO-1 AdamW: flat parameter buffer, reduce-scatter grads, sharded state, all-gather params
  checkpoint.py sharded atomic checkpoints with SHA-256 manifest and LATEST pointer
  data.py       char-level corpus and deterministic per-(step, micro-step) global batches
  train.py      training loop (single / ddp / zero), LR schedule, logging, fault injection
  launch.py     elastic launcher: store, N workers, restart-all on failure
```

## Communication

* **Rendezvous.** The launcher starts a key/value store over TCP. Each rank listens on an ephemeral port and publishes it under `a<attempt>/addr/<rank>`. Rank j connects to every rank i < j, giving a full mesh. A restart uses a fresh namespace.
* **Transport.** One TCP connection per pair of ranks, with `TCP_NODELAY` and 4 MiB socket buffers.
  * Each connection has a dedicated sender thread, so a rank can send to its ring successor while it receives from its predecessor.
  * Messages up to 64 KiB, which always fit in the socket buffer, are sent inline to avoid a thread hand-off.
  * Every message carries `(sequence number, byte count)`. A mismatch raises `CommError("protocol desync")`; data is never silently misinterpreted.
* **Ring all-reduce.**
  * Reduce-scatter: W−1 steps; afterwards rank r holds the total of chunk r+1.
  * All-gather: W−1 steps, starting from chunk r+1.
  * Chunk boundaries come from `partition(n, W)`, shared with ZeRO.
  * The summation order is fixed, so every rank gets bit-identical results and runs are reproducible.
  * The public `reduce_scatter` adds one rotation step so that rank r owns chunk r, which is what ZeRO expects.
* **Direct all-reduce.** Every rank sends its full buffer to every other rank and sums in rank order. This is the baseline: one step, but (W−1)× the data.
* **Failure detection.** Every receive has a timeout (`DTRAIN_TIMEOUT`, default 60 s). A closed connection or a timeout raises `CommError`, the worker exits non-zero, and the launcher restarts the group.

## Data parallelism

* **DDP.**
  * Parameters are broadcast from rank 0.
  * Parameters are packed into buckets of about 1 MiB, in reverse definition order.
  * The autograd engine fires a hook when a parameter's gradient is final, i.e. after all of its uses have been back-propagated. The hook copies the gradient into its bucket.
  * A complete bucket is all-reduced on the communication thread (overlap), or all buckets are reduced after backward (no overlap).
  * `finish()` waits for all buckets, divides by W and writes the averaged gradients back.
  * `no_sync()` lets gradient-accumulation micro-steps accumulate locally.
* **ZeRO-1.**
  * Parameters live in one flat buffer, and the model's arrays are views into it.
  * Gradients are flattened and reduce-scattered.
  * Each rank computes the squared norm of its own shard; one scalar all-reduce of these gives the global norm for clipping.
  * AdamW runs on the local shard only, so each rank holds 1/W of the optimizer state.
  * The updated shards are all-gathered back into the flat buffer.

## Determinism and fault tolerance

* The global batch for each (step, micro-step) is a pure function of the seed. Each rank takes its slice, a single process can reproduce a W-rank job's batch (`--emulate-world`), and a resumed run sees exactly the data it would have seen.
* **Checkpoints.**
  * Each rank writes its own shard: parameters and optimizer state, or its ZeRO shard of the optimizer state.
  * After a barrier, rank 0 writes a manifest with the SHA-256 of every shard and then atomically replaces `LATEST`.
  * On load, the world size and checksum are verified.
* **Restart.** On any worker failure the launcher stops the whole group and restarts it, up to `--max-restarts` times. Workers resume from `LATEST`. Because the data and the reductions are deterministic, the final parameters are bit-identical to a run without failures (tested).
