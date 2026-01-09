import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from soap_jax import soap


class MLP(eqx.Module):
    layers: list

    def __init__(self, key: jax.Array, in_dim: int, hidden_sizes: tuple[int, ...], out_dim: int):
        keys = jax.random.split(key, len(hidden_sizes) + 1)

        dims = [in_dim] + list(hidden_sizes)
        self.layers = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.layers.append(eqx.nn.Linear(d_in, d_out, key=keys[i]))
        self.layers.append(eqx.nn.Linear(dims[-1], out_dim, key=keys[-1]))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


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


def compute_loss(model: MLP, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    preds = jax.vmap(model)(x)
    return jnp.mean((preds - y) ** 2)


@eqx.filter_jit
def train_step(
    model: MLP, opt_state: optax.OptState, x: jnp.ndarray, y: jnp.ndarray, optimizer: optax.GradientTransformation
) -> tuple[MLP, optax.OptState, jnp.ndarray]:
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, model)  # ty:ignore[invalid-argument-type]
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def main() -> None:
    key = jax.random.PRNGKey(0)
    key, model_key = jax.random.split(key)

    x, y = make_data(key)

    model = MLP(model_key, in_dim=x.shape[-1], hidden_sizes=(256, 256, 256, 256), out_dim=y.shape[-1])
    optimizer = soap(
        learning_rate=3e-3,
        precondition_frequency=5,
        precondition_1d=False,
        weight_decay=0.01,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    initial_loss = compute_loss(model, x, y)

    for _ in range(200):
        model, opt_state, _ = train_step(model, opt_state, x, y, optimizer)
        loss = compute_loss(model, x, y)
        print(f"loss={float(loss):.6f}")

    final_loss = compute_loss(model, x, y)

    initial_val = float(initial_loss)
    final_val = float(final_loss)

    bound = 0.3
    print(f"initial_loss={initial_val:.6f} final_loss={final_val:.6f}")
    if not final_val < initial_val * bound:
        raise AssertionError(
            f"Expected loss to drop by at least {1 - bound:.0%}, but got {1 - final_val / initial_val:.2%}"
        )


if __name__ == "__main__":
    main()
