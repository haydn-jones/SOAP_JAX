# SOAP_JAX

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
I did not implement merging of dimensions. Optionally preconditioning 1D parameters is supported via `precondition_1d`. When `precondition_1d=False`, 1D parameters still follow the global SOAP init step, so they see a one-step update lag compared to Adam. I'll gladly take PR's improving other parts of the implementation as well.

The runs I've done with this implementation have gotten pretty good results so I expect that what I've done here is correct, but as always with unofficial implementations, review the code if you're going to do something important.
