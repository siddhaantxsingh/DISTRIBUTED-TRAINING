"""Finite-difference gradient checks for every op and for the full model (float64)."""
import numpy as np
import pytest

from dtrain.autograd import Tensor, causal_attention, cross_entropy, embedding, layer_norm
from dtrain.nn import GPT, GPTConfig

rng = np.random.default_rng(0)


def check(fn, *shapes, eps=1e-6, tol=1e-5):
    xs = [Tensor(rng.standard_normal(s), requires_grad=True) for s in shapes]
    out = fn(*xs)
    w = rng.standard_normal(out.shape)
    (out * Tensor(w)).sum().backward()
    for x in xs:
        num = np.zeros_like(x.data)
        it = np.nditer(x.data, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = x.data[i]
            x.data[i] = old + eps
            fp = (fn(*xs).data * w).sum()
            x.data[i] = old - eps
            fm = (fn(*xs).data * w).sum()
            x.data[i] = old
            num[i] = (fp - fm) / (2 * eps)
        np.testing.assert_allclose(x.grad, num, rtol=tol, atol=tol)


def test_elementwise_and_broadcast():
    check(lambda a, b: a + b, (3, 4), (4,))
    check(lambda a, b: a * b, (2, 3, 4), (1, 4))
    check(lambda a, b: a - b, (3, 1), (3, 5))


def test_matmul_batched_and_broadcast():
    check(lambda a, b: a @ b, (2, 3, 4), (4, 5))
    check(lambda a, b: a @ b, (2, 3, 4), (2, 4, 2))


def test_reductions_reshape_transpose():
    check(lambda a: a.sum(axis=1), (3, 4, 2))
    check(lambda a: a.mean(axis=-1, keepdims=True), (3, 4))
    check(lambda a: a.reshape(4, 6).transpose(1, 0), (2, 3, 4))


def test_activations():
    check(lambda a: a.gelu(), (4, 5))
    check(lambda a: a.relu(), (4, 5))


def test_layer_norm():
    check(lambda x, w, b: layer_norm(x, w, b), (2, 3, 6), (6,), (6,))


def test_attention():
    check(lambda q, k, v: causal_attention(q, k, v), (1, 2, 4, 3), (1, 2, 4, 3), (1, 2, 4, 3))


def test_embedding_with_repeated_indices():
    idx = np.array([[0, 2, 2], [1, 0, 2]])
    check(lambda w: embedding(w, idx), (3, 4))


def test_cross_entropy():
    t = rng.integers(0, 5, size=(2, 3))
    xs = Tensor(rng.standard_normal((2, 3, 5)), requires_grad=True)
    loss = cross_entropy(xs, t)
    loss.backward()
    num = np.zeros_like(xs.data)
    for i in np.ndindex(xs.shape):
        old = xs.data[i]
        xs.data[i] = old + 1e-6
        fp = cross_entropy(xs, t).data
        xs.data[i] = old - 1e-6
        fm = cross_entropy(xs, t).data
        xs.data[i] = old
        num[i] = (fp - fm) / 2e-6
    np.testing.assert_allclose(xs.grad, num, rtol=1e-6, atol=1e-8)


def test_full_model_gradient_float64():
    cfg = GPTConfig(vocab=7, ctx=5, dim=8, heads=2, layers=2, dtype="float64")
    m = GPT(cfg, seed=1)
    x = rng.integers(0, 7, size=(2, 5))
    y = rng.integers(0, 7, size=(2, 5))
    _, loss = m(x, y)
    loss.backward()
    for name, p in m.named_parameters():
        # spot-check a few coordinates of every parameter
        for _ in range(3):
            i = tuple(rng.integers(0, s) for s in p.shape)
            old = p.data[i]
            p.data[i] = old + 1e-6
            fp = m(x, y)[1].data
            p.data[i] = old - 1e-6
            fm = m(x, y)[1].data
            p.data[i] = old
            assert abs(p.grad[i] - (fp - fm) / 2e-6) < 1e-6, name


def test_grad_ready_hook_fires_once_with_final_gradient():
    w = Tensor(rng.standard_normal((3, 3)), requires_grad=True)
    seen = []
    w.register_grad_ready_hook(lambda t: seen.append(t.grad.copy()))
    x = Tensor(rng.standard_normal((2, 3)))
    y = (x @ w) + (x @ w) * 2.0  # w used twice
    y.sum().backward()
    assert len(seen) == 1
    np.testing.assert_allclose(seen[0], w.grad)
