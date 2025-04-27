from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1],weight_decay=1)
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Model_CNN(Layer):

    def __init__(self, input_shape=(1, 28, 28), num_classes=10, lambda_list=None):
        """
        初始化CNN模型
        Args:
            input_shape: 输入数据形状 (channels, height, width)
            num_classes: 分类数量
            lambda_list: L2正则化参数列表
        """
        super().__init__()
        self.layers = []
        self.input_shape = input_shape
        self.num_classes = num_classes

        # 第一卷积层: 1x28x28 -> 6x24x24 (kernel=5x5)
        conv1 = Conv2D(in_channels=input_shape[0], out_channels=6,
                      kernel_size=5, stride=1, padding=0)
        if lambda_list is not None:
            conv1.weight_decay = True
            conv1.weight_decay_lambda = lambda_list[0]
        relu1 = ReLU()

        # 第一池化层: 6x24x24 -> 6x12x12 (kernel=2x2)
        pool1 = MaxPool2D(kernel_size=2)

        # 第二卷积层: 6x12x12 -> 16x8x8 (kernel=5x5)
        conv2 = Conv2D(in_channels=6, out_channels=16,
                      kernel_size=5, stride=1, padding=0)
        if lambda_list is not None:
            conv2.weight_decay = True
            conv2.weight_decay_lambda = lambda_list[1]
        relu2 = ReLU()

        # 第二池化层: 16x8x8 -> 16x4x4 (kernel=2x2)
        pool2 = MaxPool2D(kernel_size=2)

        # 展平层: 16x4x4 -> 256
        self.flatten = Flatten()
        self.flatten_dim = 16 * 4 * 4

        # 第一全连接层: 256 -> 120
        fc1 = Linear(in_dim=self.flatten_dim, out_dim=120)
        if lambda_list is not None:
            fc1.weight_decay = True
            fc1.weight_decay_lambda = lambda_list[2]
        relu3 = ReLU()

        # 第二全连接层: 120 -> 84
        fc2 = Linear(in_dim=120, out_dim=84)
        if lambda_list is not None:
            fc2.weight_decay = True
            fc2.weight_decay_lambda = lambda_list[3]
        relu4 = ReLU()

        # 输出层: 84 -> 10
        fc3 = Linear(in_dim=84, out_dim=num_classes)
        if lambda_list is not None and len(lambda_list) > 4:
            fc3.weight_decay = True
            fc3.weight_decay_lambda = lambda_list[4]

        # 按顺序组合所有层
        self.layers = [
            conv1, relu1, pool1,
            conv2, relu2, pool2,
            self.flatten,
            fc1, relu3,
            fc2, relu4,
            fc3
        ]

        # 损失函数
        self.loss_fn = MultiCrossEntropyLoss(model=self, max_classes=num_classes)

    # 保持其他方法不变...
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    # 其他方法保持不变...