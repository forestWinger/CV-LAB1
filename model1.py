# 构建一个具有两个隐藏层的三层神经网络

import utils
import numpy as np
import json
import os
import shutil
import time

class Model:
    def __init__(self, hidden_layers=(128, 32), activation=('relu', 'relu', 'softmax'), input_size=784,
                 output_size=10, lambda1=0.0005):
        self.config = {'hidden_layers': hidden_layers, 'activation': activation, 'lambda1': lambda1}
        self.layers = []
        for i in range(3):
            if i == 0:
                self.layers.append(utils.Linear(input_size, hidden_layers[i]))
            else:
                if i == 1:
                    self.layers.append(utils.Linear(hidden_layers[i - 1], hidden_layers[i]))
                else:
                    self.layers.append(utils.Linear(hidden_layers[i - 1], output_size))

            self.layers.append(utils.Activation(activation[i]))

        self.lambda1 = lambda1
        self.loss = utils.SoftmaxCrossEntropyLossWithL2(self.lambda1)

        #  存储全连接层的输出值。
        self.wx = []  
        # 存储激活值。
        self.ax = []  

        # 存储测试集的全连接层的输出值
        self.wx1 = []
        # 存储测试集的激活层的输出值
        self.ax1 = []

    def refresh(self):  # 清空存储的用于计算导数的中间值
        self.ax = []
        self.wx = []

    def forward(self, x):
        self.refresh()
        self.ax.append(x)
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
            if i % 2:
                self.ax.append(x)
            else:
                self.wx.append(x)
        if self.ax:
             self.ax.pop()  
        return x

    def forward1(self, x):
        self.wx1 = []
        self.ax1 = []
        self.ax1.append(x)
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
            if i % 2:
                self.ax1.append(x)
            else:
                self.wx1.append(x)
        if self.ax1:
             self.ax1.pop()  
        return x

    def predict(self, x):
        x = self.forward(x)
        return np.argmax(x, axis=-1)

    def backward(self, x, y):
        # 存储每一层的梯度
        grad_b = [0, 0, 0]
        prob = self.forward(x)
        for i in range(len(self.layers))[-1::-2]:
            if i == len(self.layers) - 1:
                grad_b[i//2] = self.loss.backward(prob, y)
            else:
                grad_b[i//2] = np.dot(grad_b[i // 2 + 1], self.layers[i + 1].weights.T) * self.layers[i].backward(self.wx[i // 2])
        return grad_b, prob


    def update(self, x, y, learningrate):
        grad_b, prob = self.backward(x, y)
        loss = self.loss.forward(self.wx[2], y, [0, 0, 0])
        
        for i in range(len(self.layers))[::2]:
            grad_w = np.dot(self.ax[i // 2].T, np.atleast_2d(grad_b[i // 2])) + self.lambda1 * self.layers[i].weights
            self.layers[i].weights = self.layers[i].weights - learningrate * grad_w
            self.layers[i].bias = self.layers[i].bias - learningrate * np.sum(grad_b[i // 2], axis=0)
        return loss

    def test(self, X, Y):
        pred = []
        for i in range(len(X)):
            pred.append(self.predict(X[i]))
        return utils.accuracy(pred, Y)

    # 获取验证集的loss值
    def valid_loss(self, X_valid, Y_valid):
        m = self.forward1(X_valid)
        loss = self.loss.forward(self.wx1[2], Y_valid, [1, 1, 1])
        return loss




    # 获取神经网络模型中每一层的参数（权重和偏置）
    def parameter1(self):
        parameter1 = [0, 0, 0]
        for i in range(len(self.layers))[0::2]:
            parameter1[i//2] = {}
            parameter1[i//2]['weights'] = self.layers[i].weights
            parameter1[i//2]['bias'] = self.layers[i].bias
        return parameter1



    def train(self, X, Y, X_test, Y_test, learningrate=0.05, decay_number=0.9999, batch_size=32, epochs=40, start_up=200, min_lr=1e-4,
                save=False, result_dir='result/test'):
        start_time = time.time()  # 记录开始时间

        trainlosses = []
        validlosses = []
        learningrates = []

        # 保存验证集准确度
        validaccuracy_nums = []

        max_validaccuracy_num = 0
        for epoch in range(epochs):
            indices = np.random.choice(len(X), len(X), replace=True)
            print("epoch",epoch+1)
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X = X[batch_indices]
                batch_Y = Y[batch_indices]
                trainloss = self.update(batch_X, batch_Y, learningrate)
                if save:
                    validloss = self.valid_loss(X_test, Y_test)

                if i // batch_size % 10 == 0:
                    learningrate = utils.lr_decay(learningrate, epoch * len(X) + i, decay_number=decay_number, start_up=start_up, min_lr=min_lr)
                    if save:
                        learningrates.append(learningrate)
                        trainlosses.append(trainloss)
                        validlosses.append(trainloss)
                    
                if save and i // batch_size % 1000 == 0:
                        validaccuracy_num = self.test(X_test, Y_test)
                        validaccuracy_nums.append(validaccuracy_num)
                        if validaccuracy_num > max_validaccuracy_num:
                            parameter1 = self.parameter1()
                            config = self.config
                            max_validaccuracy_num = validaccuracy_num

                        print(f"Epoch: {epoch + 1}\tStep: {i}\tTrainloss: {trainloss:.15f} \tValidloss: {validloss:.15f}\tAccuracy: {validaccuracy_num:.15f}")

        # 训练完的准确率
        validaccuracy_num = self.test(X_test, Y_test)
        validaccuracy_nums.append(validaccuracy_num)
        print("验证集最高准确率: {:.15f}".format(max(validaccuracy_num, max_validaccuracy_num)))
        if save and validaccuracy_num > max_validaccuracy_num:
                    parameter1 = self.parameter1()
                    config = self.config

        # 保存 loss 和 val_acc 和最佳模型
        if save:
            try:
                if os.path.exists(result_dir):
                    shutil.rmtree(result_dir)
                os.makedirs(result_dir, exist_ok=True)
                with open(os.path.join(result_dir, 'learningrates.json'), 'w') as f:
                    json.dump(learningrates, f)
                with open(os.path.join(result_dir, 'trainloss.json'), 'w') as f:
                    json.dump(trainlosses, f)
                with open(os.path.join(result_dir, 'validlosses.json'), 'w') as f:
                    json.dump(validlosses, f)
                with open(os.path.join(result_dir, 'val_accuracy.json'), 'w') as f:
                    json.dump(validaccuracy_nums, f)
                np.save(os.path.join(result_dir, 'model.npy'), {'parameter1': parameter1, 'config': config})
                print("保存成功！")
            except Exception as e:
                print("保存失败:", e)

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算总共的运行时间
        print("Total execution time:", elapsed_time, "seconds")


def load_model(model_path):
    para_config = np.load(model_path, allow_pickle=True).item()
    config = para_config['config']
    parameter1 = para_config['parameter1']
    model = Model(hidden_layers=config['hidden_layers'], activation=config['activation'], lambda1=config['lambda1'])
    for i in range(len(parameter1) * 2)[::2]:
        model.layers[i].weights = parameter1[i // 2]['weights']
        model.layers[i].bias = parameter1[i // 2]['bias']
    return model
