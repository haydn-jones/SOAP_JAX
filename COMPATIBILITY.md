# PhiJAX compatibility

This fork starts at SOAP_JAX `v0.2.2`
(`a7c92f174c49b13136294a94ae0d46266b4f671f`). It preserves the parameter
PyTree at the Optax boundary, including NNX variable metadata and Equinox
static fields. PhiJAX needs no Trainer adapter: initialize SOAP with the
original parameters, supply the original gradients, and use `optax.apply_updates`.

## Why use an internal array tuple?

The baseline unwraps NNX variables only during initialization. Gradients still
contain `Param` nodes, so the first direct Optax update fails with a PyTree type
mismatch. `tests/test_nnx.py::test_direct_optax[True]` reproduces it; the array
control passes.

Removing that unwrapping alone fixes direct Optax, including preconditioner
refreshes. However, it breaks `nnx.Optimizer`: NNX converts custom
preconditioners inside variable values into `State` mappings, losing their
`.matrices` interface. Replacing the class with a named tuple alone also fails.

The fix therefore flattens array leaves consistently at initialization and
update, then reconstructs updates with the incoming gradient tree definition.
Internal moments and preconditioners have a stable tuple structure. Custom
preconditioners stay outside NNX variable values. No Flax runtime import or
special handling in PhiJAX is needed.

PR #3 (`9bd476938981af7e4cc8826a9984c753b45e5863`) was inspected separately.
Its tuple/mapping preconditioner conversion is unnecessary with this boundary
conversion. None of its dtype or counter changes are included. The existing
preconditioner class, arithmetic, schedule indexing, decay, bias correction,
zero first step, and refresh timing are unchanged from `v0.2.2`.
On the pinned Flax version the unmodified baseline's `nnx.Optimizer` test passes;
the unwrapping-only candidate exposes the separate failure described above.

## Checkpoints

**Optimizer-state layout changes:** moments, `GG`, and `Q` now use tuples in
JAX leaf order for all parameter representations. Existing `v0.2.2` full-state
checkpoints are not generally compatible. Load model weights and initialize a
new optimizer, or migrate each old optimizer field to the corresponding tuple
explicitly. Restarting optimizer state does not exactly resume an old run.
There is no automatic checkpoint migration in this focused change.

New checkpoints restore with PhiJAX's existing `OrbaxCheckpointIO` and a fresh
matching target state. Keep the same model structure, SOAP configuration, and
dependency versions when resuming. Exact continuation is checked against
uninterrupted training on each backend; CPU/GPU cross-backend equality is not
claimed.

## Reproduce validation

The committed `uv.lock` pins the environment. Python 3.12+ is required for the
development group; library Python support is unchanged. Core test versions:
JAX/JAXlib 0.11.1, Flax 0.12.9, Optax 0.2.8, Orbax 0.12.4, Equinox 0.13.8.

```bash
uv sync --locked --python 3.12
git show v0.2.2:src/soap_jax/soap.py > /tmp/soap-v022.py
JAX_PLATFORMS=cpu SOAP_BASELINE=/tmp/soap-v022.py uv run --no-sync pytest tests/test_nnx.py tests/test_parity.py
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
```

To run the failing reproduction against the untouched baseline:

```bash
mkdir -p /tmp/soap-baseline/soap_jax
cp /tmp/soap-v022.py /tmp/soap-baseline/soap_jax/soap.py
cp src/soap_jax/__init__.py /tmp/soap-baseline/soap_jax/__init__.py
JAX_PLATFORMS=cpu PYTHONPATH=/tmp/soap-baseline uv run --no-sync pytest tests/test_nnx.py -k direct_optax
```

The regression suite compares identical initial values and supplied gradients
across dictionaries, NNX, Equinox, and the independent baseline implementation.
It covers matrix/vector/scalar parameters, both `precondition_1d` settings,
excluded dimensions, float32/float64, finite states, exact update trees,
zero initialization updates, nine ordinary steps, four refreshes, schedules,
and weight decay. The NNX optimizer wrapper has a separate parity test.

For actual PhiJAX integration, use checkout
`6b4ba66f1587498ec778ef766f9cb9291141b1a4` (version 0.2.0b4):

```bash
git clone https://github.com/HangJung97/PhiJAX /tmp/phijax-soap-validation
git -C /tmp/phijax-soap-validation checkout 6b4ba66f1587498ec778ef766f9cb9291141b1a4
uv pip install --python .venv/bin/python /tmp/phijax-soap-validation
JAX_PLATFORMS=cpu PHIJAX_SOURCE=/tmp/phijax-soap-validation uv run --no-sync pytest tests/test_phijax.py
```

The MLP uses PhiJAX's actual compiled `make_train_step`; it saves after seven
steps and compares every state and metric for eight further steps after restore.
The heat test uses the quickstart's equation, DataModule, 32x32 MLP, and Trainer;
it compares 16 uninterrupted steps with seven saved plus nine resumed steps.
The refresh frequency is two. All TrainState fields, including optimizer slots
and random keys, must match exactly. This is a compatibility smoke test, not a
claim of converged PDE accuracy.

For CUDA 13, sync the optional group **before** installing the integration
checkout, then repeat the tests with `JAX_PLATFORMS=cuda`:

```bash
uv sync --locked --group cuda13
uv pip install --python .venv/bin/python /tmp/phijax-soap-validation
JAX_PLATFORMS=cuda PHIJAX_SOURCE=/tmp/phijax-soap-validation SOAP_BASELINE=/tmp/soap-v022.py uv run --no-sync pytest tests
```

## Validated results

All 22 tests pass on CPU and on an NVIDIA RTX PRO 2000 Blackwell Generation
Laptop GPU (driver 596.53), with the locked JAX CUDA 13 environment. Both
backends pass exact baseline/representation comparisons and full-state
checkpoint continuation. GPU validation does not establish multi-GPU support.
