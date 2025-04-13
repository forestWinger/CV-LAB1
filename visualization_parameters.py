import download_use_data
import matplotlib.pyplot as plt
import os
import model1
import numpy as np

X_train, y_train = download_use_data.load_fashion_mnist('data/fashion', kind='train')
# 获取当前脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path0 = os.path.join(script_dir, 'result/pictures')
os.makedirs(save_path0, exist_ok=True)

save_path1 = os.path.join(save_path0, 'orign.png')

# 展示 MNIST 数据集中的前 64 张图像
for i, image1 in enumerate(X_train[:64]):
    image1 = image1.reshape(28, 28)
    plt.subplot(8, 8, i+1)
    plt.imshow(image1, cmap='gray')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

plt.savefig(save_path1)
plt.show()

model_path = os.path.join(script_dir, 'result/best_model/model.npy')
best_model = model1.load_model(model_path)

# 获取模型的参数
parameters = best_model.parameter1()

# 可视化第一层权重
layer1_weights = parameters[0]['weights']

fig, axes = plt.subplots(4, 8, figsize=(16, 8))  # 调整为合适的子图网格大小
vmin, vmax = layer1_weights.min(), layer1_weights.max()  # 使用权重的最小值和最大值来设置颜色范围
for coef, ax in zip(layer1_weights.T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)  # 可视化权重
    ax.set_xticks(())  # 去除坐标轴
    ax.set_yticks(())

plt.suptitle('Visualization of Layer 1 Weights')

save_path1 = os.path.join(save_path0, 'layer1_weights_visualization.png')
plt.savefig(save_path1)
plt.show()




# 绘制权重分布图
plt.figure(figsize=(10, 5))
plt.title('Layer1-weights')
plt.hist(layer1_weights.flatten(), bins=100)
plt.xlabel('Value')
plt.ylabel('Frequency')
save_path2 = os.path.join(save_path0, 'layer1_weights_histogram.png')
plt.savefig(save_path2)
plt.show()


# 可视化第一层偏置
layer1_biases = parameters[0]['bias']
plt.figure(figsize=(10, 5))
plt.title('Layer1-biases')
plt.hist(layer1_biases.flatten(), bins=100)
plt.xlabel('Value')
plt.ylabel('Frequency')
save_path3 = os.path.join(save_path0, 'layer1_biases_histogram.png')
plt.savefig(save_path3)
plt.show()

# 可视化第二层权重
layer2_weights = parameters[1]['weights']
plt.figure(figsize=(10, 5))
plt.title('Layer2-weights')
plt.hist(layer2_weights.flatten(), bins=100)
plt.xlabel('Value')
plt.ylabel('Frequency')
save_path4 = os.path.join(save_path0, 'layer2_weights_histogram.png')
plt.savefig(save_path4)
plt.show()

# 可视化第二层偏置
layer2_biases = parameters[1]['bias']
plt.figure(figsize=(10, 5))
plt.title('Layer2-biases')
plt.hist(layer2_biases.flatten(), bins=100)
plt.xlabel('Value')
plt.ylabel('Frequency')
save_path5 = os.path.join(save_path0, 'layer2_biases_histogram.png')
plt.savefig(save_path5)
plt.show()

# 可视化第三层权重
layer3_weights = parameters[2]['weights']
plt.figure(figsize=(10, 5))
plt.title('Layer3-weights')
plt.hist(layer3_weights.flatten(), bins=100)
plt.xlabel('Value')
plt.ylabel('Frequency')
save_path6 = os.path.join(save_path0, 'layer3_weights_histogram.png')
plt.savefig(save_path6)
plt.show()

# 可视化第三层偏置
layer3_biases = parameters[2]['bias']
plt.figure(figsize=(10, 5))
plt.title('Layer3-biases')
plt.hist(layer3_biases.flatten(), bins=100)
plt.xlabel('Value')
plt.ylabel('Frequency')
save_path7 = os.path.join(save_path0, 'layer3_biases_histogram.png')
plt.savefig(save_path7)
plt.show()

