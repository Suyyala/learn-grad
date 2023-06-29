# loss function
from nuron import Module


class MSELoss(Module):
    def __init__(self):
        pass
    
    def __call__(self, x, y):
        return self.forward(x, y)
    
    def forward(self, x, y):
        return sum((x_i - y_i) ** 2 for x_i, y_i in zip(x, y))