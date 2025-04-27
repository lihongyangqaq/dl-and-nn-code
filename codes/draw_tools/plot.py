# plot the score and loss
import matplotlib.pyplot as plt

colors_set = {'Kraftime': ('#E3E37D', '#968A62')}


def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]

    # 训练数据的迭代次数
    train_epochs = [i for i in range(len(runner.train_scores))]
    # 验证数据的迭代次数，每100次迭代记录一次，最后一次迭代也记录
    dev_epochs = []
    num_batches = int(len(runner.train_scores) / len(runner.dev_scores))
    for i in range(len(runner.dev_scores) - 1):
        dev_epochs.append((i + 1) * num_batches - 1)
    dev_epochs.append(len(runner.train_scores) - 1)

    # 绘制训练损失变化曲线
    axes[0].plot(train_epochs, runner.train_loss, color=train_color, label="Train loss")
    # 绘制评价损失变化曲线
    axes[0].plot(dev_epochs, runner.dev_loss, color=dev_color, linestyle="--", label="Dev loss")
    # 绘制坐标轴和图例
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("iteration")
    axes[0].set_title("")
    axes[0].legend(loc='upper right')
    # 绘制训练准确率变化曲线
    axes[1].plot(train_epochs, runner.train_scores, color=train_color, label="Train accuracy")
    # 绘制评价准确率变化曲线
    axes[1].plot(dev_epochs, runner.dev_scores, color=dev_color, linestyle="--", label="Dev accuracy")
    # 绘制坐标轴和图例
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("iteration")
    axes[1].legend(loc='lower right')
