# measure how good the predictions are
import numpy as np
from scripts.tensor import Tensor

class Loss:
    def loss(self,predicted:Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):
    # mean squared error
    def loss(self,predicted:Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)
    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2* (predicted - actual)
