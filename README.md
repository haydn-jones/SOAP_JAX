# SOAP_JAX

## Update 01/09/2026
I saw that a number of people have been using this implementation, so I took the time to ensure it was more aligned with the official implementation and corrected some potential issues with tree leaves (I had never observed the issue, but I addressed it anyway). Also added optional 1D preconditioning.

The main fix was that the EMA updates were stale on off-precondition steps. After fixing, early steps may see slower convergence (compared to previous implementation) but this aligns with the official implementation. Your runs will now be different than before and if that is an issue just pin to `v0.1.0`.

## About
This is an *unofficial* JAX implementation of the SOAP optimizer from [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321), based on the official PyTorch implementation found here https://github.com/nikhilvyas/SOAP.

You can install this with
```
pip install git+https://github.com/haydn-jones/SOAP_JAX
```

and can use it as follows:

```python
from soap_jax import soap

opt = soap(
    learning_rate=3e-3,
    b1=0.95
    b2=0.95,
    weight_decay=0.01,
    precondition_frequency=5,
    precondition_1d=False, # default is False, set to True to precondition 1D parameters as well
)
```

I've written it similarly to how optimizers in optax are defined, so you can also import `scale_by_soap` for just the gradient transformation.

## JAX Specific Information
I did not implement merging of dimensions. I'll gladly take PR's improving other parts of the implementation as well.

## PhiJAX / NNX compatibility

This fork preserves parameter PyTrees through direct Optax updates and
`nnx.Optimizer`. See [compatibility and checkpoint migration notes](COMPATIBILITY.md)
for the pinned test environment, numerical parity tests, and PhiJAX validation.
The optimizer-state layout changes; existing full optimizer checkpoints require
migration or reinitialization.
