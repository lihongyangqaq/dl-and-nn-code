from abc import abstractmethod
import numpy as np


class Layer():
    def __init__(self) -> None:
        self.optimizable = True  # 标记该层是否可优化（是否有可训练参数）

    @abstractmethod
    def forward():
        """前向传播抽象方法"""
        pass

    @abstractmethod
    def backward():
        """反向传播抽象方法"""
        pass


class Linear(Layer):
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal,
                 weight_decay=True, weight_decay_lambda=1e-4):
        super().__init__()
        # 初始化权重
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))

        # 梯度存储
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None

        # L2正则化配置
        self.weight_decay = weight_decay
        if weight_decay:
            self.l2_reg = L2Regularization(lambda_reg=weight_decay_lambda)
        else:
            self.l2_reg = None

    def forward(self, X):
        """前向传播"""
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad):
        """反向传播"""
        # 计算普通梯度
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)

        # 添加L2正则化梯度
        if self.weight_decay:
            reg_grads = self.l2_reg.backward({'W': self.W})
            self.grads['W'] += reg_grads['W']

        return grad @ self.W.T

    def l2_penalty(self):
        """计算L2正则化损失"""
        if self.weight_decay:
            return self.l2_reg.forward({'W': self.W})  # 只对W正则化
        return 0.0

    def clear_grad(self):
        """清空梯度"""
        self.grads = {'W': None, 'b': None}

    def __call__(self, X):
        return self.forward(X)

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 weight_decay=False, weight_decay_lambda=1e-8):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.optimizable = True

        # Xavier initialization
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        self.W = np.random.normal(0, scale, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros(out_channels)  # 偏置初始化为0
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.status = {'train': True}  # Added status dictionary

    def __call__(self, X):
        return self.forward(X)

    def __str__(self):
        return f"A Conv2d Layer with fan_in:{self.in_channels}, fan_out:{self.out_channels}, kernel_size:{self.kernel_size}"

    def forward(self, X):
        batch_size, in_channels, in_height, in_width = X.shape
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Pad input if needed
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                              mode='constant')
        else:
            X_padded = X

        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # Vectorized implementation using reshape and matmul
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                # Get the current window and reshape for matmul
                window = X_padded[:, :, h_start:h_end, w_start:w_end]
                window_reshaped = window.reshape(batch_size, -1)  # [batch_size, in_channels*kernel_size*kernel_size]

                # Reshape weights and perform matmul
                W_reshaped = self.W.reshape(self.out_channels,
                                            -1).T  # [in_channels*kernel_size*kernel_size, out_channels]
                output[:, :, i, j] = np.matmul(window_reshaped, W_reshaped) + self.b

        if self.status.get('train', False):
            self.input = X_padded

        return output

    def backward(self, grads):
        batch_size, out_channels, out_height, out_width = grads.shape
        _, in_channels, in_height, in_width = self.input.shape if self.input is not None else (None, None, None, None)

        # Initialize gradients
        grad_input_padded = np.zeros_like(self.input) if self.input is not None else None
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                # Get the current window
                window = self.input[:, :, h_start:h_end, w_start:w_end]
                window_reshaped = window.reshape(batch_size, -1)  # [batch_size, in_channels*kernel_size*kernel_size]

                # Compute gradients
                current_grad = grads[:, :, i, j]  # [batch_size, out_channels]

                # Weight gradient
                grad_W += np.matmul(
                    current_grad.T,  # [out_channels, batch_size]
                    window_reshaped  # [batch_size, in_channels*kernel_size*kernel_size]
                ).reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

                # Bias gradient (average over batch)
                grad_b += current_grad.mean(axis=0)

                # Input gradient
                if grad_input_padded is not None:
                    grad_input_padded[:, :, h_start:h_end, w_start:w_end] += np.matmul(
                        current_grad,  # [batch_size, out_channels]
                        self.W.reshape(self.out_channels, -1)  # [out_channels, in_channels*kernel_size*kernel_size]
                    ).reshape(batch_size, self.in_channels, self.kernel_size, self.kernel_size)

        # Normalize gradients by spatial dimensions and batch size
        normalization_factor = out_height * out_width * batch_size
        grad_W /= normalization_factor
        grad_b /= out_height * out_width  # Bias is typically normalized by spatial dimensions only

        # Apply weight decay if needed
        if self.weight_decay:
            grad_W += 2 * self.weight_decay_lambda * self.W

        self.grads = {'W': grad_W, 'b': grad_b}

        # Remove padding from input gradient if needed
        if grad_input_padded is not None and self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        return grad_input / normalization_factor if grad_input is not None else None

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}



class MaxPool2D(Layer):
    """
    Max Pooling Layer
    """

    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.inputshape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def __str__(self):
        return f"A Max Pooling with kernel size:{self.kernel_size}"

    def forward(self, X):
        """
        (F,W,H) -> (F,W//k,H//k)
        """
        B, C, H, W = self.inputshape = X.shape
        H_out, W_out = H // self.kernel_size, W // self.kernel_size
        output = np.zeros((B, C, H_out, W_out))

        self.max_row = np.zeros((B, C, H_out, W_out), dtype=int)
        self.max_col = np.zeros((B, C, H_out, W_out), dtype=int)

        for i in range(0, H - self.kernel_size + 1, self.kernel_size):
            for j in range(0, W - self.kernel_size + 1, self.kernel_size):
                window = X[:, :, i:i + self.kernel_size, j:j + self.kernel_size]
                flat_window = window.reshape(B, C, -1)
                max_idx = np.argmax(flat_window, axis=-1)
                di = max_idx // self.kernel_size
                dj = max_idx % self.kernel_size
                self.max_row[:, :, i // self.kernel_size, j // self.kernel_size] = di
                self.max_col[:, :, i // self.kernel_size, j // self.kernel_size] = dj
                output[:, :, i // self.kernel_size, j // self.kernel_size] = np.max(window, axis=(-2, -1))

        return output

    def backward(self, grads):
        """
        (B,C,W//k,H//k) -> (B,C,W,H)
        """
        grad_X = np.zeros(self.inputshape)
        n, c, i_out, j_out = np.indices(grads.shape)
        row_indices = i_out * self.kernel_size + self.max_row
        col_indices = j_out * self.kernel_size + self.max_col
        grad_X[n, c, row_indices, col_indices] = grads

        return grad_X


class ReLU(Layer):
    """
    激活函数层。
    An activation layer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input = None  # 保存输入用于反向传播
        self.optimizable = False  # 无可训练参数

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """ReLU前向传播: max(0, x)"""
        self.input = X
        return np.where(X < 0, 0, X)  # 小于0输出0，否则输出原值

    def backward(self, grads):
        """ReLU反向传播: 输入小于0时梯度为0，否则传递原梯度"""
        assert self.input.shape == grads.shape
        return np.where(self.input < 0, 0, grads)

class Flatten(Layer):
    """
    Flatten层，将多维输入展平为一维向量。
    The Flatten layer, which flattens the multi-dimensional input into a one-dimensional vector.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None  # 保存输入的形状
        self.optimizable = False  # 无可训练参数

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        前向传播：将输入展平为一维向量
        Input X: [batch_size, C, H, W] 或其他形状
        Output: [batch_size, C * H * W]
        """
        self.input_shape = X.shape  # 保存输入形状
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)  # 展平操作

    def backward(self, grad):
        """
        反向传播：将梯度恢复为输入的形状
        Input grad: [batch_size, C * H * W]
        Output: [batch_size, C, H, W] 或其他原始形状
        """
        return grad.reshape(self.input_shape)  # 将梯度恢复为输入的形状

    def clear_grad(self):
        pass  # 无可训练参数，无需清空梯度

class MultiCrossEntropyLoss(Layer):
    """
    多分类交叉熵损失层，包含Softmax层，可以通过cancel_softmax方法取消Softmax
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """

    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model  # 关联的模型
        self.max_classes = max_classes  # 最大类别数
        self.has_softmax = True  # 默认包含softmax
        self.input = None  # 保存输入
        self.labels = None  # 保存标签
        self.grads = None  # 保存梯度
        self.optimizable = False  # 无可训练参数

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        前向传播计算损失
        predicts: [batch_size, D] 预测值
        labels : [batch_size, ] 真实标签
        """
        self.input = predicts
        self.labels = labels

        if self.has_softmax:
            # 如果有softmax，先计算softmax
            probs = softmax(predicts)
        else:
            # 否则直接使用输入作为概率(假设输入已经是概率)
            probs = predicts

        # 确保数值稳定性
        probs = np.clip(probs, 1e-15, 1 - 1e-15)

        # 获取每个样本正确类别的概率
        batch_size = predicts.shape[0]
        correct_probs = probs[np.arange(batch_size), labels]

        # 计算交叉熵损失
        loss = -np.mean(np.log(correct_probs))

        # 保存softmax输出用于反向传播
        self.probs = probs
        return loss

    def backward(self):
        """反向传播计算梯度"""
        batch_size = self.input.shape[0]

        if self.has_softmax:
            # 如果有softmax，计算softmax的梯度
            grad = self.probs.copy()
            grad[np.arange(batch_size), self.labels] -= 1
            grad /= batch_size
        else:
            # 否则直接使用输入梯度
            grad = self.input.copy()
            grad[np.arange(batch_size), self.labels] -= 1
            grad /= batch_size

        self.grads = grad
        # 将梯度传递给模型进行反向传播
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        """取消softmax层"""
        self.has_softmax = False
        return self


class L2Regularization(Layer):
    """
    L2正则化层，可以作为权重衰减在Linear类中实现。
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """

    def __init__(self, lambda_reg=1e-4) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg  # 正则化系数
        self.optimizable = False  # 无可训练参数

    def forward(self, params):
        """
        计算L2正则化损失
        params: 参数字典
        """
        l2_loss = 0
        for key in params:
            l2_loss += 0.5 * self.lambda_reg * np.sum(params[key] ** 2)
        return l2_loss

    def backward(self, params):
        """
        计算L2正则化梯度
        params: 参数字典
        """
        grads = {}
        for key in params:
            grads[key] = self.lambda_reg * params[key]
        return grads


def softmax(X):
    """softmax函数"""
    x_max = np.max(X, axis=1, keepdims=True)  # 每行最大值(防止数值溢出)
    x_exp = np.exp(X - x_max)  # 指数运算
    partition = np.sum(x_exp, axis=1, keepdims=True)  # 归一化分母
    return x_exp / partition  # softmax结果