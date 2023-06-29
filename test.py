from nn import SequentialModel
from nuron import Layer


fcn = SequentialModel(layers=[Layer(2, 25), Layer(25, 25), Layer(25, 1)])
print(fcn)

# test sequential model using test data
test_x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
test_y = [0, 1, 1, 0]

y = fcn(test_x)
print(y)



