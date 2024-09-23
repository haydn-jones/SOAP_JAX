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
)
```

I've written it similarly to how optimizers in optax are defined, so you can also import `scale_by_soap` for just the gradient transformation.

## JAX Specific Information
I did not implement merging of dimensions or optionally preconditioning <2D parameters. I'll gladly take PR's implementing these features, they just weren't necessary for me. Further, this is the first time I've implemented an optimizer in JAX so I'd be happy to take PR's improving its implementation as well.

The runs I've done with this implementation have gotten pretty good results so I expect that what I've done here is correct, but as always with unofficial implementations, review the code if you're going to do something important.