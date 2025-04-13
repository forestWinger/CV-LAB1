import download_use_data
import model1
import matplotlib.pyplot as plt
import os

# 找到最好的学习率
def find_best_learningrate():
    print("正在寻找最好的学习率")
    learningrates=[0.5,0.4,0.3, 0.25, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]
    accuracy_nums = []
    for learningratei in learningrates:
        model_lr = model1.Model(lambda1=0.01, hidden_layers=(784, 256))
        model_lr.train(X_train1, y_train1, X_valid, y_valid, batch_size=32, decay_number=0.9999, epochs=18, learningrate=learningratei,
                       start_up=250, min_lr=0.0001)
        accuracy_num = model_lr.test(X_test, y_test)
        accuracy_nums.append(accuracy_num)
        print("accuracy_num",accuracy_num)
    img = plt.figure(1)
    x = range(len(learningrates))
    plt.plot(x, accuracy_nums, marker='o')
    plt.xlabel('learning rate')
    plt.ylabel('the accuracy of test')
    plt.xticks(x, learningrates)
    plt.savefig(save_path1)
    print("最好的学习率:",learningrates[accuracy_nums.index(max(accuracy_nums))])
    return learningrates[accuracy_nums.index(max(accuracy_nums))]

# 找到最好的正则化参数
def find_best_lambda(learningrate):
    
    print("正在寻找最好的正则化参数")
    lambda1s=[1, 0.5,0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    accuracy_nums = []
    for lambda1 in lambda1s:
        model_lambda = model1.Model(lambda1=lambda1, hidden_layers=(512, 256))
        model_lambda.train(X_train1, y_train1, X_valid, y_valid, batch_size=32, decay_number=0.9999, epochs=18, learningrate=learningrate,
                           start_up=250, min_lr=0.0001)
        accuracy_num = model_lambda.test(X_test, y_test)
        accuracy_nums.append(accuracy_num)

    img = plt.figure(2)
    x = range(len(lambda1s))
    plt.plot(x, accuracy_nums, marker='o')
    plt.xlabel('regularization parameter')
    plt.ylabel('the accuracy of test')
    plt.xticks(x, lambda1s)
    plt.savefig(save_path2)
    print("最好的正则化参数:",lambda1s[accuracy_nums.index(max(accuracy_nums))])
    return lambda1s[accuracy_nums.index(max(accuracy_nums))]


def find_best_hidden_layer(i=0, learningrate=0.01, lambda1=0.01, hidden_layers=[784, 512, 256, 128, 64, 32, 16, 8], another_hidden_layers=256):
    accuracy_nums = []
    print(f"正在寻找最好的隐藏层{i}的大小")
    for hidden_layer in hidden_layers:
        h = [another_hidden_layers, another_hidden_layers]
        h[i] = hidden_layer
        model_hidden = model1.Model(lambda1=lambda1, hidden_layers=tuple(h))
        model_hidden.train(X_train1, y_train1, X_valid, y_valid, batch_size=32, decay_number=0.9999, epochs=18, learningrate=learningrate,
                           start_up=250, min_lr=0.0001)
        accuracy_num = model_hidden.test(X_test, y_test)
        accuracy_nums.append(accuracy_num)

    img = plt.figure(2 + i +1)
    x = range(len(hidden_layers))
    plt.plot(x, accuracy_nums, marker='o')
    plt.xlabel('hidden layer size')
    plt.ylabel('the accuracy of test')
    plt.xticks(x, hidden_layers)
    save_path3 = os.path.join(script_dir, 'result/pictures/find_hidden' + str(i) + '_accuracy' + '.png')
    plt.savefig(save_path3)
    print(f"最好的隐藏层{i}的大小: {hidden_layers[accuracy_nums.index(max(accuracy_nums))]}")
    return hidden_layers[accuracy_nums.index(max(accuracy_nums))]


def find_best_activation(learningrate=0.01,lambda1=0.0001,hidden_layers=(784,256)):
    print("正在寻找最好的激活函数")
    accuracy_nums = []
    activations=[('relu', 'sigmoid', 'softmax'),('relu', 'tanh', 'softmax'),('relu', 'relu', 'softmax'),
                ('sigmoid', 'relu', 'softmax'),('sigmoid', 'sigmoid', 'softmax'),('sigmoid', 'tanh', 'softmax'),
                ('tanh', 'relu', 'softmax'),('tanh', 'sigmoid', 'softmax'),('tanh', 'tanh', 'softmax')]
    for activation in activations:
        model_act = model1.Model(lambda1=lambda1, hidden_layers=hidden_layers,activation=activation)
        model_act.train(X_train1, y_train1, X_valid, y_valid, batch_size=32, decay_number=0.9999, epochs=18, learningrate=learningrate,
                       start_up=250, min_lr=0.0001)
        accuracy_num = model_act.test(X_test, y_test)
        accuracy_nums.append(accuracy_num)
        print("accuracy_num",accuracy_num)
    img = plt.figure(5)
    x = range(len(activations))
    plt.plot(x, accuracy_nums, marker='o')
    plt.xlabel('kind')
    plt.ylabel('the accuracy of test')
    kinds = [i for i in range(1, 10)]
    plt.xticks(x, kinds)
    plt.savefig(save_path5)
    print("最好的激活函数:",activations[accuracy_nums.index(max(accuracy_nums))])
    return activations[accuracy_nums.index(max(accuracy_nums))]


if __name__ == '__main__':
    
    X_train, y_train = download_use_data.load_fashion_mnist('data/fashion', kind='train')
    X_test, y_test = download_use_data.load_fashion_mnist('data/fashion', kind='t10k')

    # 划分数据集比例
    train_ratio = 0.8
    validation_ratio = 0.2

    # 计算划分的数据数量
    total_samples = len(X_train)
    train_samples = int(total_samples * train_ratio)

    # 划分训练集和验证集
    X_train1, y_train1 = X_train[:train_samples], y_train[:train_samples]
    X_valid, y_valid = X_train[train_samples:], y_train[train_samples:]


    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构造保存文件的路径（相对于脚本所在的目录）
    save_path0 = os.path.join(script_dir, 'result/pictures')
    save_path1 = os.path.join(script_dir, 'result/pictures/find_best_learning_rate.png')
    save_path2 = os.path.join(script_dir, 'result/pictures/find_best_regularization_parameter.png')
    save_path4 = os.path.join(script_dir, 'result/best_model')
    save_path5 = os.path.join(script_dir, 'result/pictures/find_best_activation.png')

    os.makedirs(save_path0, exist_ok=True)




    learningrate = find_best_learningrate()
    lambda1 = find_best_lambda(learningrate=learningrate)
    h1 = find_best_hidden_layer(i=0, learningrate=learningrate, lambda1=lambda1,  another_hidden_layers=256)
    h2 = find_best_hidden_layer(i=1, learningrate=learningrate, lambda1=lambda1,  another_hidden_layers=h1)
    activation = find_best_activation(learningrate=learningrate, lambda1=lambda1, hidden_layers=(h1,h2))

    best_model = model1.Model(lambda1=lambda1, hidden_layers=(h1, h2),activation=activation)

    # 保存最好的模型
    best_model.train(X_train1, y_train1, X_valid, y_valid, learningrate=learningrate, batch_size=32, decay_number=0.9999, epochs=30, start_up=250, min_lr=0.0001, save=True, result_dir=save_path4)

