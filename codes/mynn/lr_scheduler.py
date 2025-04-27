from abc import abstractmethod
import numpy as np

# 定义调度器基类，它是一个抽象类，不能直接实例化，用于规范所有学习率调度器的基本结构
class Scheduler:
    def __init__(self, optimizer) -> None:
        # 接收一个优化器对象作为参数，后续会通过这个优化器来调整学习率
        self.optimizer = optimizer
        # 用于记录当前执行的步数，初始化为 0
        self.step_count = 0

    @abstractmethod
    def step(self):
        """
        抽象方法，用于执行学习率的调整操作。
        所有继承自 Scheduler 的子类都必须实现这个方法。
        """
        pass

# 定义 StepLR 类，继承自 Scheduler 类，实现按固定步数调整学习率的功能
class StepLR(Scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        # 调用父类的构造函数，初始化优化器和步数计数器
        super().__init__(optimizer)
        # 每经过 step_size 步，学习率会进行一次调整
        self.step_size = step_size
        # 学习率调整的因子，即每次调整时学习率乘以 gamma
        self.gamma = gamma

    def step(self) -> None:
        # 每调用一次 step 方法，步数计数器加 1
        self.step_count += 1
        # 当步数达到或超过 step_size 时，进行学习率调整
        if self.step_count >= self.step_size:
            # 将优化器的初始学习率乘以 gamma，实现学习率的调整
            self.optimizer.init_lr *= self.gamma
            # 调整后，将步数计数器重置为 0，以便下一轮的调整
            self.step_count = 0

class MultiStepLR(Scheduler):
    def __init__(self, optimizer, milestones=[30, 60],gamma=0.1 ) -> None:
        super().__init__(optimizer)
        self.milestones = milestones
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count in self.milestones:
            self.optimizer.init_lr *= self.gamma


# 定义 ExponentialLR 类，继承自 Scheduler 类，实现按指数衰减调整学习率的功能
class ExponentialLR(Scheduler):
    def __init__(self, optimizer, gamma=0.95) -> None:
        # 调用父类的构造函数，初始化优化器和步数计数器
        super().__init__(optimizer)
        # 学习率衰减的因子，每次调整时学习率乘以 gamma
        self.gamma = gamma

    def step(self) -> None:
        # 每调用一次 step 方法，步数计数器加 1
        self.step_count += 1
        # 每次调用 step 方法，都将优化器的初始学习率乘以 gamma，实现指数衰减
        self.optimizer.init_lr *= self.gamma