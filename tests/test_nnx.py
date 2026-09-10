import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from soap_jax import soap


class Parameters(nnx.Module):
    def __init__(self):
        self.matrix = nnx.Param(jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 10)
        self.vector = nnx.Param(jnp.ones(3))
        self.scalar = nnx.Param(jnp.asarray(0.5))


@pytest.mark.parametrize("wrapped", [False, True])
def test_direct_optax(wrapped):
    graph, params = nnx.split(Parameters())
    if not wrapped:
        params = nnx.to_pure_dict(params)
    tx = soap(precondition_frequency=2)
    state = tx.init(params)

    @jax.jit
    def step(params, state):
        def loss(p):
            if wrapped:
                model = nnx.merge(graph, p)
                return sum(jnp.sum(v**2) for v in (model.matrix[...], model.vector[...], model.scalar[...]))
            return sum(jnp.sum(v**2) for v in jax.tree.leaves(p))

        value, grads = jax.value_and_grad(loss)(params)
        updates, state = tx.update(grads, state, params)
        assert jax.tree.structure(updates) == jax.tree.structure(params)
        return optax.apply_updates(params, updates), state, value

    for _ in range(8):
        params, state, loss = step(params, state)
        assert jnp.isfinite(loss)


@pytest.mark.parametrize("precondition_1d", [False, True])
def test_nnx_optimizer(precondition_1d):
    model = Parameters()
    tx = soap(precondition_frequency=2, precondition_1d=precondition_1d)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    reference = nnx.to_pure_dict(nnx.state(model))
    state = tx.init(reference)

    @jax.jit
    def update(grads, state, params):
        updates, state = tx.update(grads, state, params)
        return optax.apply_updates(params, updates), state

    @nnx.jit
    def step(model, optimizer, grads):
        optimizer.update(model, grads)

    for index in range(8):
        gradients = jax.tree.map(lambda p, index=index: jnp.sin(p + 0.1 * index), reference)
        wrapped = jax.tree.unflatten(jax.tree.structure(nnx.state(model)), jax.tree.leaves(gradients))
        step(model, optimizer, wrapped)
        reference, state = update(gradients, state, reference)
        for actual, expected in zip(jax.tree.leaves(nnx.state(model)), jax.tree.leaves(reference)):
            assert jnp.array_equal(actual, expected)
            assert jnp.isfinite(actual).all()
