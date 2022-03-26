# Second-Order-Derivative-in-MAML
**retain_graph：** 动态图，进行完反向传播计算图就会默认释放，如果想要保留，则设置该参数为`True`。

**create_graph：** 记录求导过程中的函数，即生成导数计算图，如果要求高阶导数，需要设置该参数为`True`。

> Second-order = False

```python
Initial parameter value:
w0:  -1.9347891807556152
b0:  0.0
Support set
w.is_leaf= True b.is_leaf= True
Gradient Over Support Set:
gradient of w0 -3.8695783615112305
gradient of b0 -3.8695783615112305
Calculate fast weights...
Fast Weights(w1): -1.9347891807556152--3.8695783615112305= 1.9347891807556152
Fast Weights(b1): 0.0--3.8695783615112305= 3.8695783615112305
Query set
w.is_leaf= False b.is_leaf= False
First-Order, delta loss/ delta w0 requires_grad ? False
First-Order, delta loss/ delta w0 is_leaf ? True
After calculating gradient on query set, let check the parameters of meta learner.
current w of meta-learner -1.9347891807556152
current b of meta-learner 0.0
(Automate) First-order approximate derivative of loss on query set w.r.t. w 30.956626892089844
Manual:
First-order derivative 30.956626892089844
After update on query set.
(Automate): 
 w0'=-32.891414642333984
(Manual): 
w0'=w0-gw1
-32.89141607284546'=-1.9347891807556152-30.956626892089844
```

> Second-order = True

```python
Initial parameter value:
w0:  -0.7597505450248718
b0:  0.0
Support set
w.is_leaf= True b.is_leaf= True
Gradient Over Support Set:
gradient of w0 -1.5195010900497437
gradient of b0 -1.5195010900497437
Calculate fast weights...
Fast Weights(w1): -0.7597505450248718--1.5195010900497437= 0.7597505450248718
Fast Weights(b1): 0.0--1.5195010900497437= 1.5195010900497437
Query set
w.is_leaf= False b.is_leaf= False
Second Order, delta loss/ delta w0 requires_grad ? True
Second Order, delta loss/ delta w0 is_leaf ? False
After calculating gradient on query set, let check the parameters of meta learner.
current w of meta-learner -0.7597505450248718
current b of meta-learner 0.0
(Automate) Second-order derivative of loss on query set w.r.t. w -24.312015533447266
Manual:
Second-order derivative -24.3120174407959
After update on query set.
(Automate): 
 w0'=23.552265167236328
(Manual): 
w0'=w0-gw2
23.552266895771027'=-0.7597505450248718--24.3120174407959
```



## Computation Graph

![image-20220326144318653](https://github.com/tdye24/Second-Order-Derivative-in-MAML/blob/master/computation%20graph.jpg)
