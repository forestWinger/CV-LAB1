import download_use_data
import model1
import os

#最好的学习率: 0.01最好的正则化参数: 0.0001,最好的隐藏层0的大小: 784,最好的隐藏层0的大小: 256

# 获取当前脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'result/best_model/model.npy')

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
best_model = model1.load_model(model_path)

train_accuracy=best_model.test(X_test, y_test)
valid_accuracy=best_model.test(X_train, y_train)
test_accuracy=best_model.test(X_valid, y_valid)


print(f"训练集准确率: {train_accuracy:.4%}")
print(f"验证集准确率: {valid_accuracy:.4%}")
print(f"测试集准确率: {test_accuracy:.4%}")
