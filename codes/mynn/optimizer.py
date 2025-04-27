from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass

class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        #print(f"Number of layers in the model: {len(self.model.layers)}")
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]

class MomentGD(Optimizer):
    def __init__(self, init_lr, model, beta=0.9):
        super().__init__(init_lr, model)
        self.beta = beta
        # 为每个可优化层的每个参数初始化上一时刻的动量（初始为0）
        self.momentum = {}
        for layer in self.model.layers:
            if layer.optimizable:
                self.momentum[layer] = {}
                for key in layer.params.keys():
                    self.momentum[layer][key] = 0

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    # 计算动量更新
                    self.momentum[layer][key] = self.beta * self.momentum[layer][key] + self.init_lr * layer.grads[key]
                    # 根据动量更新参数
                    layer.params[key] -= self.momentum[layer][key]