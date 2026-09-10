import importlib.util
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from soap_jax import soap


class EqxParameters(eqx.Module):
    matrix: jax.Array
    scalar: jax.Array
    vector: jax.Array
    label: str = eqx.field(static=True, default="control")


def assert_arrays_equal(left, right):
    a, b = jax.tree.leaves(left), jax.tree.leaves(right)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)
        assert np.isfinite(x).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("precondition_1d", [False, True])
@pytest.mark.parametrize("scheduled,decay", [(False, 0.0), (True, 0.1)])
@pytest.mark.parametrize("max_dim", [2, 10])
def test_equivalent_supplied_gradients(dtype, precondition_1d, scheduled, decay, max_dim):
    arrays = dict(
        matrix=jnp.arange(6, dtype=dtype).reshape(2, 3) / 10,
        scalar=jnp.asarray(0.5, dtype),
        vector=jnp.arange(3, dtype=dtype) / 5,
    )
    model = nnx.Module()
    for key, value in arrays.items():
        setattr(model, key, nnx.Param(value, tag="preserve-metadata"))
    _, wrapped = nnx.split(model)
    params = [arrays, wrapped, EqxParameters(**arrays)]
    rate = optax.exponential_decay(0.003, 3, 0.8) if scheduled else 0.003
    kwargs = dict(
        learning_rate=rate,
        weight_decay=decay,
        precondition_frequency=2,
        precondition_1d=precondition_1d,
        max_precond_dim=max_dim,
    )
    tx = soap(**kwargs)
    transforms = [tx] * 3
    # Optional independent v0.2.2 oracle, extracted with git show (see compatibility.md).
    if baseline_path := os.environ.get("SOAP_BASELINE"):
        spec = importlib.util.spec_from_file_location("soap_baseline", baseline_path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        transforms.append(baseline.soap(**kwargs))
        params.append(arrays)
    states = [t.init(p) for t, p in zip(transforms, params)]
    structures = [jax.tree.structure(s) for s in states]
    updates_fns = [jax.jit(t.update) for t in transforms]
    for step in range(10):
        supplied = jax.tree.map(lambda p, step=step: jnp.sin(p + (step + 1) * 0.17).astype(dtype), arrays)
        all_updates = []
        for i, (p, state, update) in enumerate(zip(params, states, updates_fns)):
            gradients = jax.tree.unflatten(jax.tree.structure(p), jax.tree.leaves(supplied))
            u, states[i] = update(gradients, state, p)
            assert jax.tree.structure(u) == jax.tree.structure(p)
            assert jax.tree.structure(states[i]) == structures[i]
            for leaf in jax.tree.leaves(states[i]):
                assert np.isfinite(leaf).all()
                if jnp.issubdtype(leaf.dtype, jnp.floating):
                    assert leaf.dtype == dtype
            if step == 0:
                for leaf in jax.tree.leaves(u):
                    np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))
            params[i] = optax.apply_updates(p, u)
            all_updates.append(u)
        for i in range(1, len(params)):
            assert_arrays_equal(all_updates[0], all_updates[i])
            assert_arrays_equal(params[0], params[i])
            assert_arrays_equal(states[0], states[i])
        if step == 1:
            assert any(np.any(x != 0) for x in jax.tree.leaves(all_updates[0]))
