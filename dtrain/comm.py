"""Process groups and collectives over TCP, from scratch.

* `Store`: a tiny TCP key/value rendezvous server (`StoreServer`) and client.
  Ranks publish their listening address under `addr/<rank>`.
* `ProcessGroup`: a full mesh of TCP connections. Each peer connection has
  a dedicated sender thread, so a rank can send to its ring successor while
  receiving from its predecessor (full duplex, no deadlock). Every message
  carries a small header (collective sequence number, byte count) so a
  desynchronised peer is detected instead of silently corrupting data.
* Collectives on contiguous NumPy arrays: ring all-reduce (reduce-scatter
  then all-gather, 2(W-1)/W x the data sent per rank), a direct all-reduce baseline
  (every rank sends everything to everyone), reduce-scatter, all-gather,
  broadcast and barrier. Ring reductions use a fixed order, so results are
  bit-identical between runs with the same world size.
* Async: `allreduce_async` queues work on a single communication thread
  (FIFO, so all ranks issue collectives in the same order) and returns a
  handle with `wait()`. DDP uses it to overlap gradient reduction with the
  rest of the backward pass.
* Failure handling: every blocking receive has a timeout; a closed or
  timed-out peer raises `CommError`. The elastic launcher restarts the job.
"""
from __future__ import annotations

import os
import queue
import socket
import socketserver
import struct
import threading
import time

import numpy as np


class CommError(RuntimeError):
    pass


# ------------------------------------------------------------------ store
class _StoreHandler(socketserver.StreamRequestHandler):
    def handle(self):
        srv = self.server
        while True:
            hdr = self.rfile.read(9)
            if len(hdr) < 9:
                return
            op, klen, vlen = struct.unpack("<BII", hdr)
            key = self.rfile.read(klen).decode()
            val = self.rfile.read(vlen)
            with srv.cv:
                if op == 1:  # set
                    srv.kv[key] = val
                    srv.cv.notify_all()
                    out = b""
                elif op == 2:  # get (blocking)
                    while key not in srv.kv:
                        srv.cv.wait()
                    out = srv.kv[key]
                elif op == 3:  # add
                    cur = int(srv.kv.get(key, b"0")) + int(val)
                    srv.kv[key] = str(cur).encode()
                    srv.cv.notify_all()
                    out = srv.kv[key]
                else:
                    return
            self.wfile.write(struct.pack("<I", len(out)) + out)
            self.wfile.flush()


class StoreServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host="127.0.0.1", port=0):
        super().__init__((host, port), _StoreHandler)
        self.kv = {}
        self.cv = threading.Condition()
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def address(self):
        return f"{self.server_address[0]}:{self.server_address[1]}"

    def close(self):
        self.shutdown()
        self.server_close()


class Store:
    def __init__(self, address: str, timeout: float = 60.0):
        host, port = address.rsplit(":", 1)
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.sock = socket.create_connection((host, int(port)), timeout=timeout)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise CommError(f"cannot reach store at {address}")
                time.sleep(0.05)
        self.f = self.sock.makefile("rwb")

    def _req(self, op, key, val=b""):
        k = key.encode()
        self.f.write(struct.pack("<BII", op, len(k), len(val)) + k + val)
        self.f.flush()
        hdr = self.f.read(4)
        if len(hdr) < 4:
            raise CommError("store connection closed")
        (n,) = struct.unpack("<I", hdr)
        return self.f.read(n)

    def set(self, key, val: bytes):
        self._req(1, key, val)

    def get(self, key) -> bytes:
        return self._req(2, key)

    def add(self, key, n: int) -> int:
        return int(self._req(3, key, str(n).encode()))

    def close(self):
        self.sock.close()


# ------------------------------------------------------------ peer channel
_HDR = struct.Struct("<QQ")  # sequence number, payload bytes
INLINE_SEND_BYTES = 64 << 10


class _Peer:
    def __init__(self, sock: socket.socket, timeout: float):
        self.sock = sock
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
            try:
                sock.setsockopt(socket.SOL_SOCKET, opt, 4 << 20)
            except OSError:
                pass
        sock.settimeout(timeout)
        self.q: queue.Queue = queue.Queue()
        self.err: Exception | None = None
        self.t = threading.Thread(target=self._sender, daemon=True)
        self.t.start()

    def _sender(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            seq, mv, done = item
            try:
                self.sock.sendall(_HDR.pack(seq, mv.nbytes))
                if mv.nbytes:
                    self.sock.sendall(mv)
            except Exception as e:  # noqa: BLE001 - reported to the waiting caller
                self.err = e
            done.set()

    def send(self, seq: int, arr: np.ndarray) -> threading.Event:
        done = threading.Event()
        mv = memoryview(np.ascontiguousarray(arr)).cast("B")
        if mv.nbytes <= INLINE_SEND_BYTES and self.q.empty():
            # Small enough to fit in the kernel socket buffer (configured to
            # 4 MiB): sendall returns without waiting for the peer to read,
            # so sending inline cannot deadlock and skips a thread hand-off.
            try:
                self.sock.sendall(_HDR.pack(seq, mv.nbytes) + mv.tobytes())
            except Exception as e:  # noqa: BLE001
                self.err = e
            done.set()
            return done
        self.q.put((seq, mv, done))
        return done

    def recv_into(self, seq: int, out: np.ndarray):
        hdr = self._recv_exact(_HDR.size)
        rseq, n = _HDR.unpack(hdr)
        if rseq != seq or n != out.nbytes:
            raise CommError(f"protocol desync: expected seq {seq} / {out.nbytes} bytes, got seq {rseq} / {n} bytes")
        if n:
            mv = memoryview(out).cast("B")
            got = 0
            while got < n:
                try:
                    k = self.sock.recv_into(mv[got:], n - got)
                except socket.timeout as e:
                    raise CommError("timed out waiting for a peer") from e
                except OSError as e:
                    raise CommError(f"peer connection failed: {e}") from e
                if k == 0:
                    raise CommError("peer closed the connection")
                got += k

    def _recv_exact(self, n):
        buf = bytearray(n)
        mv = memoryview(buf)
        got = 0
        while got < n:
            try:
                k = self.sock.recv_into(mv[got:], n - got)
            except socket.timeout as e:
                raise CommError("timed out waiting for a peer") from e
            except OSError as e:
                raise CommError(f"peer connection failed: {e}") from e
            if k == 0:
                raise CommError("peer closed the connection")
            got += k
        return bytes(buf)

    def close(self):
        self.q.put(None)
        try:
            self.sock.close()
        except OSError:
            pass


def partition(n: int, world: int):
    """Chunk boundaries used by ring collectives and ZeRO: rank r owns [b[r], b[r+1])."""
    base, extra = divmod(n, world)
    b = [0]
    for r in range(world):
        b.append(b[-1] + base + (1 if r < extra else 0))
    return b


# ----------------------------------------------------------- process group
class Work:
    def __init__(self):
        self._ev = threading.Event()
        self._err = None

    def wait(self):
        self._ev.wait()
        if self._err:
            raise self._err


class ProcessGroup:
    def __init__(self, rank: int, world: int, store_addr: str, timeout: float = 60.0, prefix: str = ""):
        self.rank, self.world, self.timeout = rank, world, timeout
        self.seq = 0
        self.bytes_sent = 0
        self.peers: dict[int, _Peer] = {}
        self.store = Store(store_addr, timeout)
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(world)
        lsock.settimeout(timeout)
        self.store.set(f"{prefix}addr/{rank}", f"127.0.0.1:{lsock.getsockname()[1]}".encode())
        # rank j > i connects to i; i accepts from every j > i
        for i in range(rank):
            host, port = self.store.get(f"{prefix}addr/{i}").decode().rsplit(":", 1)
            s = socket.create_connection((host, int(port)), timeout=timeout)
            s.sendall(struct.pack("<I", rank))
            self.peers[i] = _Peer(s, timeout)
        for _ in range(world - rank - 1):
            try:
                s, _ = lsock.accept()
            except socket.timeout as e:
                raise CommError("timed out waiting for peers to connect") from e
            s.settimeout(timeout)
            (j,) = struct.unpack("<I", _recv_n(s, 4))
            self.peers[j] = _Peer(s, timeout)
        lsock.close()
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self.barrier()

    # -------------------------------------------------------- primitives
    def _next_seq(self):
        self.seq += 1
        return self.seq

    def _wait_send(self, peer: _Peer, ev: threading.Event):
        if not ev.wait(self.timeout):
            raise CommError("timed out sending to a peer")
        if peer.err:
            raise CommError(f"send failed: {peer.err}")

    def _sendrecv(self, to: int, send_arr: np.ndarray, frm: int, recv_out: np.ndarray, seq: int):
        ps = self.peers[to]
        ev = ps.send(seq, send_arr)
        self.bytes_sent += send_arr.nbytes
        self.peers[frm].recv_into(seq, recv_out)
        self._wait_send(ps, ev)

    # -------------------------------------------------------- collectives
    def allreduce(self, arr: np.ndarray, algo: str = "ring") -> np.ndarray:
        """In-place sum over all ranks. Returns `arr`."""
        if self.world == 1:
            return arr
        if not arr.flags.c_contiguous:
            raise ValueError("allreduce needs a contiguous array")
        flat = arr.reshape(-1)
        if algo == "ring":
            # after reduce-scatter (without the ownership rotation) rank r
            # holds the total of chunk r+1; gather from there
            self._ring_reduce_scatter(flat, rotate=False)
            self._ring_all_gather(flat, offset=1)
        elif algo == "direct":
            self._direct_allreduce(flat)
        else:
            raise ValueError(algo)
        return arr

    def reduce_scatter(self, flat: np.ndarray) -> np.ndarray:
        """Sum over ranks; afterwards this rank's partition of `flat` holds the
        total (other partitions hold partial sums). Returns a view of it."""
        if self.world > 1:
            self._ring_reduce_scatter(flat)
        b = partition(flat.size, self.world)
        return flat[b[self.rank]:b[self.rank + 1]]

    def all_gather(self, flat: np.ndarray) -> np.ndarray:
        """Each rank's partition of `flat` is copied to every rank."""
        if self.world > 1:
            self._ring_all_gather(flat)
        return flat

    def broadcast(self, arr: np.ndarray, root: int = 0) -> np.ndarray:
        if self.world == 1:
            return arr
        seq = self._next_seq()
        flat = arr.reshape(-1)
        if self.rank == root:
            evs = [(self.peers[r], self.peers[r].send(seq, flat)) for r in range(self.world) if r != root]
            for p, ev in evs:
                self._wait_send(p, ev)
            self.bytes_sent += flat.nbytes * (self.world - 1)
        else:
            self.peers[root].recv_into(seq, flat)
        return arr

    def barrier(self):
        x = np.zeros(1, np.float64)
        self.allreduce(x)

    def _ring_reduce_scatter(self, flat: np.ndarray, rotate: bool = True):
        W, r = self.world, self.rank
        b = partition(flat.size, W)
        nxt, prv = (r + 1) % W, (r - 1) % W
        tmp = np.empty(max(b[i + 1] - b[i] for i in range(W)), flat.dtype)
        for s in range(W - 1):
            seq = self._next_seq()
            si, ri = (r - s) % W, (r - s - 1) % W
            send = flat[b[si]:b[si + 1]]
            recv = tmp[: b[ri + 1] - b[ri]]
            self._sendrecv(nxt, send, prv, recv, seq)
            flat[b[ri]:b[ri + 1]] += recv
        # after W-1 steps rank r holds the full sum of chunk (r + 1) % W;
        # rotate ownership so that rank r owns chunk r (one more hop)
        if not rotate:
            return
        seq = self._next_seq()
        own = (r + 1) % W
        mine = np.empty(b[r + 1] - b[r], flat.dtype)
        self._sendrecv(nxt, flat[b[own]:b[own + 1]], prv, mine, seq)
        flat[b[r]:b[r + 1]] = mine

    def _ring_all_gather(self, flat: np.ndarray, offset: int = 0):
        """Rank r starts holding chunk (r + offset) % W; afterwards all chunks."""
        W, r = self.world, self.rank
        b = partition(flat.size, W)
        nxt, prv = (r + 1) % W, (r - 1) % W
        for s in range(W - 1):
            seq = self._next_seq()
            si, ri = (r + offset - s) % W, (r + offset - s - 1) % W
            self._sendrecv(nxt, flat[b[si]:b[si + 1]], prv, flat[b[ri]:b[ri + 1]], seq)

    def _direct_allreduce(self, flat: np.ndarray):
        seq = self._next_seq()
        evs = [(self.peers[p], self.peers[p].send(seq, flat.copy())) for p in range(self.world) if p != self.rank]
        self.bytes_sent += flat.nbytes * (self.world - 1)
        acc = flat.copy()
        tmp = np.empty_like(flat)
        # fixed summation order (rank 0, 1, ...) so every rank gets identical bits
        parts = {self.rank: acc}
        for p in range(self.world):
            if p != self.rank:
                self.peers[p].recv_into(seq, tmp)
                parts[p] = tmp.copy()
        for peer, ev in evs:
            self._wait_send(peer, ev)
        out = parts[0].copy()
        for p in range(1, self.world):
            out += parts[p]
        flat[...] = out

    # ------------------------------------------------------------- async
    def allreduce_async(self, arr: np.ndarray, algo: str = "ring") -> Work:
        w = Work()
        self._q.put((arr, algo, w))
        return w

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            arr, algo, w = item
            try:
                self.allreduce(arr, algo)
            except Exception as e:  # noqa: BLE001 - surfaced by Work.wait
                w._err = e
            w._ev.set()

    def close(self):
        self._q.put(None)
        for p in self.peers.values():
            p.close()
        self.store.close()


def _recv_n(s, n):
    buf = b""
    while len(buf) < n:
        k = s.recv(n - len(buf))
        if not k:
            raise CommError("peer closed during handshake")
        buf += k
    return buf


def init_from_env(timeout: float | None = None) -> ProcessGroup:
    """Create the default group from RANK / WORLD_SIZE / STORE_ADDR (set by the launcher)."""
    t = timeout if timeout is not None else float(os.environ.get("DTRAIN_TIMEOUT", "60"))
    return ProcessGroup(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), os.environ["STORE_ADDR"], t,
                        prefix=os.environ.get("DTRAIN_PREFIX", ""))
