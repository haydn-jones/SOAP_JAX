import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state

from soap_jax import soap


class MLP(nn.Module):
    hidden_sizes: tuple[int, ...]
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for width in self.hidden_sizes:
            x = nn.Dense(width)(x)
            x = nn.tanh(x)

        return nn.Dense(self.out_dim)(x)


def make_data(
    key: jax.Array,
    num_samples: int = 256,
    input_dim: int = 32,
    output_dim: int = 8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    x = jax.random.normal(k1, (num_samples, input_dim))

    true_w = jax.random.normal(k2, (input_dim, output_dim))
    true_b = jax.random.normal(k3, (output_dim,))

    noise = 0.05 * jax.random.normal(k4, (num_samples, output_dim))

    y = x @ true_w + true_b + noise
    return x, y


def create_state(
    key: jax.Array,
    model: nn.Module,
    x: jnp.ndarray,
    tx,
) -> train_state.TrainState:
    params = model.init(key, x)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def compute_loss(params: dict, apply_fn, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    preds = apply_fn(params, x)
    return jnp.mean((preds - y) ** 2)


@jax.jit
def train_step(
    state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray
) -> tuple[train_state.TrainState, jnp.ndarray]:
    def loss_fn(params: dict) -> jnp.ndarray:
        return compute_loss(params, state.apply_fn, x, y)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def main() -> None:
    key = jax.random.PRNGKey(0)

    x, y = make_data(key)

    model = MLP(hidden_sizes=(64, 64), out_dim=y.shape[-1])
    tx = soap(
        learning_rate=3e-3,
        precondition_frequency=1,
        precondition_1d=False,
    )
    state = create_state(jax.random.PRNGKey(1), model, x, tx)

    initial_loss = compute_loss(state.params, state.apply_fn, x, y)  # ty:ignore[invalid-argument-type]

    for _ in range(200):
        state, _ = train_step(state, x, y)
        loss = compute_loss(state.params, state.apply_fn, x, y)
        print(f"loss={float(loss):.6f}")

    final_loss = compute_loss(state.params, state.apply_fn, x, y)  # ty:ignore[invalid-argument-type]

    initial_val = float(initial_loss)
    final_val = float(final_loss)

    print(f"initial_loss={initial_val:.6f} final_loss={final_val:.6f}")
    if not final_val < initial_val * 0.5:
        raise AssertionError("Expected loss to drop by at least 50%.")


if __name__ == "__main__":
    main()
