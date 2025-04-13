import numpy as np

class Relu:

    def forward(self, x):
        return np.where(x > 0, x, 0)

    def backward(self, x):
        return np.where(x > 0, 1, 0)


class Sigmoid:

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y * (1 - y)


class Tanh:

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        y = np.tanh(x)
        return 1 - y**2


class Softmax:

    def forward(self, x):

        # 针对数值溢出采用的的优化方法是将每一个输出值减去输出值中最大的值
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        softmax_probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return softmax_probs

    def backward(self, x):
        x = x[0]
        x = self.forward(x)
        m = np.diag(x)
        for i in range(len(m)):
            for j in range(len(m)):
                if i == j:
                    m[i][j] = x[i] * (1 - x[i])
                else:
                    m[i][j] = -x[i] * x[j]
        return m

# 选择激活函数的类型
class Activation:

    def __init__(self, act_type):
        self.type = act_type

    def forward(self, x):
        if self.type == 'relu':
            return Relu().forward(x)
        if self.type == 'sigmoid':
            return Sigmoid().forward(x)
        if self.type == 'tanh':
            return Tanh().forward(x)
        if self.type == 'softmax':
            return Softmax().forward(x)
        if self.type == 'None':
            return x

    def backward(self, x):
        if self.type == 'relu':
            return Relu().backward(x)
        if self.type == 'sigmoid':
            return Sigmoid().backward(x)
        if self.type == 'tanh':
            return Tanh().backward(x)
        if self.type == 'softmax':
            return Softmax().backward(x)
        if self.type == 'None':
            return np.ones(x.shape)

# 标准的线性变换操作
class Linear:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weights = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros((1, out_features))

    def forward(self, x):
        x = np.dot(x, self.weights) + self.bias
        return x

# 计算交叉熵损失
class CrossEntropyLossWithL2:
    def __init__(self, lambda1=0):
        self.lambda1 = lambda1

    def forward(self, pred, label, weight):
        epsilon = 1e-6
        pred = np.clip(pred, epsilon, 1)
        l, _ = pred.shape
        loss = -np.sum(label * np.log(pred)) / l
        l2 = 0
        for i in range(len(weight)):
            l2 += 0.5 * np.sum(weight[i] ** 2)  # 累加每个权重的L2范数
        return loss + self.lambda1 * l2  # 最后再乘以 lambda1


    def backward(self, pred, label):
        epsilon = 1e-6
        pred = np.clip(pred, epsilon, 1)
        return -label / pred


class SoftmaxCrossEntropyLossWithL2:
    def __init__(self, lambda1=0):
        self.lambda1 = lambda1

    def forward(self, pred, label, weights):
        pred = Softmax().forward(pred)
        loss = CrossEntropyLossWithL2(self.lambda1).forward(pred, label, weights )
        return loss

    def backward(self, pred, label):
        l, _ = pred.shape
        return (pred - label)/l

# 学习率衰减函数
def lr_decay(learningrate, iterations, decay_number, start_up, min_lr):
    if iterations <= start_up or learningrate <= min_lr:
        return learningrate
    if iterations >= start_up and min_lr < learningrate:
        return learningrate * decay_number

# 计算准确率
def accuracy(pred, label):
    right1 = 0
    for i in range(len(pred)):
        if pred[i] == np.argmax(label[i]):
            right1 += 1
    return right1/len(pred)

