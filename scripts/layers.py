 # each layer needs to pass its inputs forward and propragate gradients backward
from typing import Dict

import numpy as np

from scripts.tensor import Tensor

class Layer:
    def __init__(self) -> None:
            self.params: Dict[str, Tensor] = {}
            self.grads: Dict[str, Tensor] = {}

    def forward(self,inputs: Tensor) -> Tensor:
        #produtce the outputs corresponding to these inputs
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        #backpropagate this gradient through the layer
        raise NotImplementedError

class Linear(Layer):
    #output  = inputs @w + b
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs: (batch_size, input_size)
        # outputs: (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)
    def forward(self, inputs: Tensor) ->Tensor:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, gradient: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(gradient, axis=0)
        self.grads["w"] = self.inputs.T @ gradient
        return gradient @ self.params["w"].T