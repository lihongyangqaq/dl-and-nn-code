import mynn as nn
from draw_tools.plot import plot
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import time
# 固定随机种子
np.random.seed(309)

# 数据加载
train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)  # 调整为CNN需要的形状 [N, C, H, W]

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# 划分验证集
idx = np.random.permutation(np.arange(num))
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# 归一化
train_imgs = train_imgs / 255.0
valid_imgs = valid_imgs / 255.0


# 实现CNN模型

# 创建模型、优化器和损失函数
cnn_model = nn.models.Model_CNN(
    input_shape=(1, 28, 28),
    num_classes=10,
    lambda_list=[1e-4,1e-4,1e-4,1e-4,1e-4]  # 对应5个可训练层
)

optimizer = nn.optimizer.SGD(init_lr=0.1, model=cnn_model)
#optimizer = nn.optimizer.MomentGD(init_lr=0.06, model=cnn_model,beta=0.9)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                        milestones=[800, 2400, 4000],
                                        gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model,
                                      max_classes=train_labs.max() + 1)
print("startrun")
# 创建训练器并训练
runner = nn.runner.RunnerM(cnn_model, optimizer,
                           nn.metric.accuracy, loss_fn,
                           scheduler=scheduler)

runner.train([train_imgs, train_labs],
             [valid_imgs, valid_labs],
             num_epochs=5,
             log_iters=10000,
             save_dir=r'./best_models')

# 绘制训练曲线
_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)
plt.show()