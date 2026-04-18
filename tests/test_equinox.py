import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from soap_jax import soap


class TinyMLP(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(self, key: jax.Array, in_dim: int, hidden_dim: int, out_dim: int):
        key1, key2 = jax.random.split(key)
        self.layers = [  # ty:ignore[invalid-assignment]
            eqx.nn.Linear(in_dim, hidden_dim, key=key1),
            eqx.nn.Linear(hidden_dim, out_dim, key=key2),
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.tanh(self.layers[0](x))
        return self.layers[1](x)


def make_data(
    key: jax.Array, num_samples: int = 64, input_dim: int = 8, output_dim: int = 3
) -> tuple[jax.Array, jax.Array]:
    key_x, key_w, key_b = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (num_samples, input_dim))
    true_w = jax.random.normal(key_w, (input_dim, output_dim))
    true_b = jax.random.normal(key_b, (output_dim,))
    y = x @ true_w + true_b
    return x, y


def compute_loss(model: TinyMLP, x: jax.Array, y: jax.Array) -> jax.Array:
    predictions = jax.vmap(model)(x)
    return jnp.mean((predictions - y) ** 2)


@eqx.filter_jit
def train_step(
    model: TinyMLP,
    opt_state: optax.OptState,
    x: jax.Array,
    y: jax.Array,
    optimizer: optax.GradientTransformation,
) -> tuple[TinyMLP, optax.OptState, jax.Array]:
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, model)  # ty:ignore[invalid-argument-type]
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def test_soap_trains_a_small_equinox_mlp() -> None:
    key = jax.random.PRNGKey(0)
    data_key, model_key = jax.random.split(key)
    x, y = make_data(data_key)

    model: TinyMLP
    model = TinyMLP(model_key, in_dim=x.shape[-1], hidden_dim=16, out_dim=y.shape[-1])  # ty:ignore[invalid-assignment]
    optimizer = soap(
        learning_rate=3e-2,
        precondition_frequency=2,
        precondition_1d=False,
        weight_decay=0.0,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    initial_loss = float(compute_loss(model, x, y))

    for _ in range(40):
        model, opt_state, _ = train_step(model, opt_state, x, y, optimizer)

    final_loss = float(compute_loss(model, x, y))

    assert final_loss < initial_loss * 0.5
