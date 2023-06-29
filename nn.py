# neural networks based on nuron.py
from nuron import Module


class SequentialModel(Module):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def backward(self):
        for layer in reversed(self.layers):
            for neuron in layer.neurons:
                neuron.backward()



