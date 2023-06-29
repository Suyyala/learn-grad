# basic building block of the network use only torch for tensors

import random

class Param:
    def __init__(self, data, children=(), op='', requires_grad=True):
        self.data = data
        self.grad = 0
        self.requires_grad = requires_grad
        # internal bookkeeping for compute graph
        self.children = children
        self.op = op
        self.backward = None
    
    def __add__(self, other):
        other = other if isinstance(other, Param) else Param(other, requires_grad=False)
        out = Param(self.data + other.data, (self, other), '+')
        
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward = backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Param) else Param(other, requires_grad=False)
        print(self.data, other.data)
        out = Param(self.data * other.data, (self, other), '*')
        
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.backward = backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Param(self.data ** other, (self,), f'**{other}')
        
        def backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out.backward = backward
        return out
    
    def __repr__(self) -> str:
        return f'data: {self.data}, grad: {self.grad}, children {self.children}'

class Module:
    def __init__(self) -> None:
        pass

    def clear_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

# basic building block of the network
class Neuron(Module):
    def __init__(self, nw):
        self.w = [Param(random.uniform(-1, 1)) for _ in range(nw)]
        self.b = Param(0)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        # this multiplication order is important as x is not a param
        return sum(w_i * x_i for w_i, x_i in zip(self.w, x)) + self.b
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self) -> str:
        return super().__repr__()

# list of neurons
class Layer(Module):
    def __init__(self, nw, no):
        self.neurons = [Neuron(nw) for _ in range(no)]

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def __repr__(self) -> str:
        return f'neurons: {self.neurons}'
    


    
    
