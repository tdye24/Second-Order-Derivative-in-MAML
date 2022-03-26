import torch
import torch.nn as nn
import torch.nn.functional as F

second_order = True


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = [
            ('linear', [1, 1])]
        self.weights = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'linear':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.weights.append(w)
                self.weights.append(nn.Parameter(torch.zeros(param[0])))

    def forward(self, xx, weights=None):
        if weights is None:
            weights = self.weights
        idx = 0
        for name, param in self.config:
            if name is 'linear':
                w, b = weights[idx], weights[idx + 1]  # w.is_leaf False, b.is_leaf False
                print("w.is_leaf=", w.is_leaf, "b.is_leaf=", b.is_leaf)
                xx = F.linear(xx, w, b)
                idx += 2
        return xx


model = Model()
meta_optimizer = torch.optim.SGD(model.parameters(), lr=1)
print("Initial parameter value:")

w0 = list(model.parameters())[0].data.item()
print("w0: ", w0)
b0 = list(model.parameters())[1].data.item()
print("b0: ", b0)

# support set
print("Support set")
x_spt = torch.tensor([1.0])
loss = model(x_spt) ** 2
if second_order:
    # 计算二阶导数，那就要构建一阶导数对应的计算图，所以这里create_graph=True
    one_order_gradients = torch.autograd.grad(outputs=loss, inputs=model.parameters(), create_graph=True)

else:
    # 采用一阶近似，那就不用构建一阶导数对应的计算图，create_graph=False (default)
    one_order_gradients = torch.autograd.grad(outputs=loss, inputs=model.parameters())
print("Gradient Over Support Set:")
print("gradient of w0", one_order_gradients[0].data.item())
print("gradient of b0", one_order_gradients[1].data.item())

print("Calculate fast weights...")
fast_weights = list(map(lambda p: p[1] - 1 * p[0], zip(one_order_gradients, model.parameters())))
# fast_weights[0] 和 fast_weights[1] 都不是叶子节点，因为 fast_weights[0] = model.parameters()[0] - one_order_gradients[0]
w1 = fast_weights[0].data.item()
b1 = fast_weights[1].data.item()
print(f"Fast Weights(w1): {w0}-{one_order_gradients[0].data.item()}=", w1)
print(f"Fast Weights(b1): {b0}-{one_order_gradients[1].data.item()}=", b1)

# query set
print("Query set")
x_qry = torch.tensor([2.0])
loss = model(x_qry, fast_weights) ** 2
if second_order:
    print("Second Order, delta loss/ delta w0 requires_grad ?", one_order_gradients[0].requires_grad)
    print("Second Order, delta loss/ delta w0 is_leaf ?", one_order_gradients[0].is_leaf)

else:
    print("First-Order, delta loss/ delta w0 requires_grad ?", one_order_gradients[0].requires_grad)
    print("First-Order, delta loss/ delta w0 is_leaf ?", one_order_gradients[0].is_leaf)

# # # #
meta_optimizer.zero_grad()
# # # #

loss.backward()

print("After calculating gradient on query set, let check the parameters of meta learner.")
print(f"current w of meta-learner {list(model.parameters())[0].data.item()}")
print(f"current b of meta-learner {list(model.parameters())[1].data.item()}")

if second_order:
    print("(Automate) Second-order derivative of loss on query set w.r.t. w", model.weights[0].grad.data.item())
else:
    print("(Automate) First-order approximate derivative of loss on query set w.r.t. w", model.weights[0].grad.data.item())

print("Manual:")

if not second_order:
    gw1 = 2 * (w1 * x_qry + b1) * x_qry
    gw1 = gw1.data.item()
    print("First-order derivative", gw1)
else:
    gw2 = 2 * (w1 * x_qry + b1) * x_qry * (1 - 2 * x_spt ** 2) + 2 * (w1 * x_qry + b1) * 1 * (- 2 * x_spt)
    gw2 = gw2.data.item()
    print("Second-order derivative", gw2)

#
#       --> w1 -->
# w0--               -->  loss on query set
#       --> b1 -->
#

meta_optimizer.step()

print("After update on query set.")
print(f"(Automate): \n w0'={list(model.parameters())[0].data.item()}")
print("(Manual): ")
if second_order:
    print(f"w0'=w0-gw2")
    print(f"{w0-gw2}'={w0}-{gw2}")
else:
    print(f"w0'=w0-gw1")
    print(f"{w0-gw1}'={w0}-{gw1}")
