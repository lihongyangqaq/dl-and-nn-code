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
            print(param_list)
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

    def save_model(self, save_path):
        """保存CNN模型参数"""
        param_list = {
            'model_type': 'CNN',  # 标识模型类型
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'flatten_dim': self.flatten_dim,
            'layers': []
        }

        for layer in self.layers:
            layer_info = {'type': layer.__class__.__name__}

            if isinstance(layer, Conv2D):
                layer_info.update({
                    'in_channels': layer.in_channels,
                    'out_channels': layer.out_channels,
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride,
                    'padding': layer.padding,
                    'params': {
                        'W': layer.params['W'],
                        'b': layer.params['b']
                    },
                    'weight_decay': getattr(layer, 'weight_decay', False),
                    'weight_decay_lambda': getattr(layer, 'weight_decay_lambda', 0)
                })
            elif isinstance(layer, Linear):
                layer_info.update({
                    'in_dim': layer.in_dim,
                    'out_dim': layer.out_dim,
                    'params': {
                        'W': layer.params['W'],
                        'b': layer.params['b']
                    },
                    'weight_decay': getattr(layer, 'weight_decay', False),
                    'weight_decay_lambda': getattr(layer, 'weight_decay_lambda', 0)
                })
            elif isinstance(layer, MaxPool2D):
                layer_info['kernel_size'] = layer.kernel_size

            param_list['layers'].append(layer_info)

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def load_model(self, load_path):
        """加载CNN模型参数"""
        with open(load_path, 'rb') as f:
            param_list = pickle.load(f)

        # 检查模型类型是否匹配
        if param_list.get('model_type') != 'CNN':
            raise ValueError("Loaded model is not a CNN model")

        # 确保所有必要键都存在
        required_keys = ['input_shape', 'num_classes', 'flatten_dim', 'layers']
        for key in required_keys:
            if key not in param_list:
                raise KeyError(f"Missing required key in model file: {key}")

        self.input_shape = param_list['input_shape']
        self.num_classes = param_list['num_classes']
        self.flatten_dim = param_list['flatten_dim']
        self.layers = []

        for layer_info in param_list['layers']:
            layer_type = layer_info['type']

            if layer_type == 'Conv2D':
                layer = Conv2D(
                    in_channels=layer_info['in_channels'],
                    out_channels=layer_info['out_channels'],
                    kernel_size=layer_info['kernel_size'],
                    stride=layer_info['stride'],
                    padding=layer_info['padding']
                )
                # 确保参数存在
                if 'params' not in layer_info:
                    raise KeyError("Missing 'params' in Conv2D layer info")
                layer.params = {
                    'W': layer_info['params']['W'],
                    'b': layer_info['params']['b']
                }
                layer.weight_decay = layer_info.get('weight_decay', False)
                layer.weight_decay_lambda = layer_info.get('weight_decay_lambda', 0)
                self.layers.append(layer)

            elif layer_type == 'Linear':
                layer = Linear(
                    in_dim=layer_info['in_dim'],
                    out_dim=layer_info['out_dim']
                )
                if 'params' not in layer_info:
                    raise KeyError("Missing 'params' in Linear layer info")
                layer.params = {
                    'W': layer_info['params']['W'],
                    'b': layer_info['params']['b']
                }
                layer.weight_decay = layer_info.get('weight_decay', False)
                layer.weight_decay_lambda = layer_info.get('weight_decay_lambda', 0)
                self.layers.append(layer)

            elif layer_type == 'ReLU':
                self.layers.append(ReLU())

            elif layer_type == 'MaxPool2D':
                # 从保存的参数中获取kernel_size，如果没有则使用默认值2
                kernel_size = layer_info.get('kernel_size', 2)
                self.layers.append(MaxPool2D(kernel_size=kernel_size))

            elif layer_type == 'Flatten':
                self.flatten = Flatten()
                self.layers.append(self.flatten)