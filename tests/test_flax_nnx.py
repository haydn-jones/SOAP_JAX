import jax
import jax.numpy as jnp
from flax import nnx

from soap_jax import soap


class TinyMLP(nnx.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.tanh(self.linear1(x))
        return self.linear2(x)


def make_data(
    key: jax.Array,
    num_samples: int = 64,
    input_dim: int = 8,
    output_dim: int = 3,
) -> tuple[jax.Array, jax.Array]:
    key_x, key_w, key_b = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (num_samples, input_dim))
    true_w = jax.random.normal(key_w, (input_dim, output_dim))
    true_b = jax.random.normal(key_b, (output_dim,))
    y = x @ true_w + true_b
    return x, y


def compute_loss(model: TinyMLP, x: jax.Array, y: jax.Array) -> jax.Array:
    predictions = model(x)
    return jnp.mean((predictions - y) ** 2)


@nnx.jit
def train_step(model: TinyMLP, optimizer: nnx.Optimizer, x: jax.Array, y: jax.Array) -> jax.Array:
    loss, grads = nnx.value_and_grad(compute_loss, argnums=nnx.DiffState(0, nnx.Param))(model, x, y)
    optimizer.update(model, grads)
    return loss


def test_soap_trains_a_small_flax_nnx_mlp() -> None:
    key = jax.random.PRNGKey(0)
    data_key, model_key = jax.random.split(key)
    x, y = make_data(data_key)

    model = TinyMLP(x.shape[-1], 16, y.shape[-1], rngs=nnx.Rngs(model_key))
    optimizer = nnx.Optimizer(
        model,
        soap(
            learning_rate=1e-2,
            precondition_frequency=2,
            precondition_1d=False,
            weight_decay=0.0,
        ),
        wrt=nnx.Param,
    )

    initial_loss = float(compute_loss(model, x, y))

    for _ in range(40):
        train_step(model, optimizer, x, y)

    final_loss = float(compute_loss(model, x, y))

    assert final_loss < initial_loss * 0.5
    assert int(optimizer.step.value) == 40
