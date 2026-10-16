"""Collective correctness across real processes (2, 3 and 4 ranks)."""
import multiprocessing as mp

import numpy as np
import pytest

from dtrain.comm import CommError, ProcessGroup, StoreServer, partition


def _run(world, fn, *args, timeout=60):
    srv = StoreServer()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_entry, args=(r, world, srv.address, fn, args, q)) for r in range(world)]
    for p in ps:
        p.start()
    out = {}
    for _ in range(world):
        r, res = q.get(timeout=timeout)
        out[r] = res
    for p in ps:
        p.join(timeout)
    srv.close()
    errs = [v for v in out.values() if isinstance(v, BaseException)]
    if errs:
        raise errs[0]
    return [out[r] for r in range(world)]


def _entry(rank, world, addr, fn, args, q):
    try:
        pg = ProcessGroup(rank, world, addr, timeout=30)
        res = fn(pg, *args)
        pg.close()
        q.put((rank, res))
    except BaseException as e:  # noqa: BLE001
        q.put((rank, e))


def _allreduce(pg, sizes, algo, dtype):
    out = []
    for i, n in enumerate(sizes):
        x = np.random.default_rng(1000 * i + pg.rank).standard_normal(n).astype(dtype)
        pg.allreduce(x, algo)
        out.append(x)
    return out


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("algo", ["ring", "direct"])
def test_allreduce_matches_numpy_sum(world, algo):
    sizes = [0, 1, 2, 3, 5, 7, 64, 1000, 12345]
    res = _run(world, _allreduce, sizes, algo, np.float64)
    for i, n in enumerate(sizes):
        expect = sum(np.random.default_rng(1000 * i + r).standard_normal(n) for r in range(world))
        for r in range(world):
            np.testing.assert_allclose(res[r][i], expect, rtol=1e-12, atol=1e-12)
            # every rank holds bit-identical results
            assert np.array_equal(res[r][i], res[0][i])


def _rs_ag(pg, n):
    flat = np.arange(n, dtype=np.float64) * (pg.rank + 1)
    mine = pg.reduce_scatter(flat).copy()
    b = partition(n, pg.world)
    full = np.zeros(n)
    full[b[pg.rank]:b[pg.rank + 1]] = mine
    pg.all_gather(full)
    return mine, full


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("n", [1, 3, 10, 1001])
def test_reduce_scatter_and_all_gather(world, n):
    res = _run(world, _rs_ag, n)
    total = np.arange(n, dtype=np.float64) * sum(range(1, world + 1))
    b = partition(n, world)
    for r, (mine, full) in enumerate(res):
        np.testing.assert_array_equal(mine, total[b[r]:b[r + 1]])
        np.testing.assert_array_equal(full, total)


def _bcast_async(pg):
    x = np.full(100, float(pg.rank))
    pg.broadcast(x, root=1)
    works = [pg.allreduce_async(np.full(10, float(i + pg.rank))) for i in range(5)]
    for w in works:
        w.wait()
    return x


def test_broadcast_and_async_order():
    res = _run(3, _bcast_async)
    for x in res:
        np.testing.assert_array_equal(x, np.ones(100))


def _dies(pg):
    if pg.rank == 1:
        import os
        os._exit(3)  # simulate a crash without closing cleanly
    x = np.ones(1000)
    try:
        pg.allreduce(x)
    except CommError as e:
        return e
    return None


def test_peer_failure_raises_commerror_not_hang():
    srv = StoreServer()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_entry, args=(r, 2, srv.address, _dies, (), q)) for r in range(2)]
    for p in ps:
        p.start()
    r, res = q.get(timeout=60)
    assert r == 0 and isinstance(res, CommError)
    for p in ps:
        p.join(10)
    srv.close()
