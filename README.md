# Second-Order-Derivative-in-MAML
> Second-order = False

```python
Initial parameter value:
w0:  Parameter containing:
tensor([[-0.5258]], requires_grad=True)
b0:  Parameter containing:
tensor([0.], requires_grad=True)
Support set
w.is_leaf= True b.is_leaf= True
Gradient Over Support Set:
gradient of w0 tensor([[-1.0516]])
gradient of b0 tensor([-1.0516])
Calculate fast weights...
Fast Weights(w1): tensor([[0.5258]], grad_fn=<SubBackward0>) tensor([[0.5258]], grad_fn=<SubBackward0>)
Fast Weights(b1): tensor([1.0516], grad_fn=<SubBackward0>) tensor([1.0516], grad_fn=<SubBackward0>)
Query set
w.is_leaf= False b.is_leaf= False
After calculating gradient on query set, let check the parameters of meta learner.
current w of meta-learner Parameter containing:
tensor([[-0.5258]], requires_grad=True)
current b of meta-learner Parameter containing:
tensor([0.], requires_grad=True)
(Automate) First-order approximate derivative of loss on query set w.r.t. w tensor([[8.4130]])
Manual:
Second-order derivative tensor([[-16.8260]], grad_fn=<AddBackward0>)
First-order derivative tensor([[8.4130]], grad_fn=<MulBackward0>)
```

> Second-order = True

```python
Initial parameter value:
w0:  Parameter containing:
tensor([[0.5523]], requires_grad=True)
b0:  Parameter containing:
tensor([0.], requires_grad=True)
Support set
w.is_leaf= True b.is_leaf= True
Gradient Over Support Set:
gradient of w0 tensor([[1.1046]], grad_fn=<TBackward>)
gradient of b0 tensor([1.1046], grad_fn=<MulBackward0>)
Calculate fast weights...
Fast Weights(w1): tensor([[-0.5523]], grad_fn=<SubBackward0>) tensor([[-0.5523]], grad_fn=<SubBackward0>)
Fast Weights(b1): tensor([-1.1046], grad_fn=<SubBackward0>) tensor([-1.1046], grad_fn=<SubBackward0>)
Query set
w.is_leaf= False b.is_leaf= False
After calculating gradient on query set, let check the parameters of meta learner.
current w of meta-learner Parameter containing:
tensor([[0.5523]], requires_grad=True)
current b of meta-learner Parameter containing:
tensor([0.], requires_grad=True)
(Automate) Second-order derivative of loss on query set w.r.t. w tensor([[17.6735]])
Manual:
Second-order derivative tensor([[17.6735]], grad_fn=<AddBackward0>)
First-order derivative tensor([[-8.8368]], grad_fn=<MulBackward0>)
```

