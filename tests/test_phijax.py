"""Optional integration tests; install PhiJAX and set PHIJAX_SOURCE for its example."""

import importlib.util
import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

pytest.importorskip("phijax")
from phijax.balancers import StaticLossBalancer
from phijax.core import BasePhiModule, PhiModule
from phijax.equations import base_data_fidelity
from phijax.models import build_mlp
from phijax.objectives import CompositeObjective
from phijax.training import OrbaxCheckpointIO, Trainer, initialize_train_state
from phijax.training.steps import make_train_step

from soap_jax import soap


class Regression(BasePhiModule):
    loss_names = ("mse",)

    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def forward(self, model_state, inputs):
        return nnx.merge(self.graph, model_state)(inputs)

    def training_step(self, model_state, batches):
        return {"mse": jnp.mean((self.forward(model_state, batches["x"]) - batches["y"]) ** 2)}


def assert_identical(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        assert a.dtype == b.dtype
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)
        assert np.isfinite(a).all()


def test_compiled_mlp_checkpoint_resume(tmp_path):
    model = nnx.Sequential(nnx.Linear(2, 8, rngs=nnx.Rngs(0)), nnx.tanh, nnx.Linear(8, 1, rngs=nnx.Rngs(1)))
    graph, params = nnx.split(model)
    module = Regression(graph)
    balancer = StaticLossBalancer(module.loss_names)
    tx = soap(
        learning_rate=optax.exponential_decay(0.003, 3, 0.9),
        weight_decay=0.01,
        precondition_frequency=2,
        precondition_1d=True,
    )
    initial = initialize_train_state(params, tx, balancer.initialize(), jax.random.key(2))
    step = make_train_step(module, balancer, tx)
    batches = {"x": jnp.asarray([[0.1, 0.2], [0.4, 0.8]], jnp.float32), "y": jnp.asarray([[0.3], [1.2]], jnp.float32)}
    state = initial
    for _ in range(7):
        state, metrics = step(state, batches)
    with OrbaxCheckpointIO(tmp_path / "mlp", enable_async_checkpointing=False) as io:
        assert io.save(state, 7)
        resumed = io.restore(initial)
    assert_identical(state, resumed)
    for _ in range(8):
        state, metrics = step(state, batches)
        resumed, restored_metrics = step(resumed, batches)
        assert_identical(state, resumed)
        assert_identical(metrics, restored_metrics)
    assert jax.tree.structure(state) == jax.tree.structure(initial)


def test_heat_example_checkpoint_resume(tmp_path):
    source = os.environ.get("PHIJAX_SOURCE")
    if not source:
        pytest.skip("Set PHIJAX_SOURCE to the validated PhiJAX checkout")
    spec = importlib.util.spec_from_file_location("heat_quickstart", Path(source) / "examples/quickstart.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    objective = CompositeObjective.from_equations(
        {"initial": base_data_fidelity, "boundary": base_data_fidelity, "pde": example.heat_equation}
    )
    module = PhiModule(
        partial(build_mlp, input_dim=2, output_dim=1, hidden=(32, 32), activation="tanh", input_norm=True),
        objective,
        name="Heat SOAP",
    )
    tx = soap(learning_rate=0.001, precondition_frequency=2)
    accelerator = "gpu" if jax.default_backend() == "gpu" else "cpu"

    def train(steps, checkpoint=None):
        trainer = Trainer(
            max_steps=steps,
            accelerator=accelerator,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=tmp_path,
        )
        return trainer.fit(module, datamodule=example.HeatDataModule(), optimizer=tx, seed=0, ckpt_path=checkpoint)

    uninterrupted = train(16)
    interrupted = train(7)
    with OrbaxCheckpointIO(tmp_path / "heat", enable_async_checkpointing=False) as io:
        assert io.save(interrupted.state, 7)
    resumed = train(9, tmp_path / "heat")
    assert int(uninterrupted.state.step) == int(resumed.state.step) == 16
    assert_identical(uninterrupted.state, resumed.state)
    assert uninterrupted.metrics == resumed.metrics
    assert all(np.isfinite(v) for v in resumed.metrics.values())
