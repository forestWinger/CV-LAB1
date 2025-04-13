import matplotlib.pyplot as plt
import json
import os

#最好的学习率: 0.001最好的正则化参数: 0.1，最好的隐藏层1的大小: 784最好的隐藏层0的大小: 256

# 获取当前脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构造保存文件的路径（相对于脚本所在的目录）
save_path1 = os.path.join(script_dir, 'result/pictures/trainloss.png')
save_path2 = os.path.join(script_dir, 'result/pictures/validloss.png')
save_path3 = os.path.join(script_dir, 'result/pictures/test_accuracy.png')
save_path4 = os.path.join(script_dir, 'result/pictures/learningrates.png')

model_path = os.path.join(script_dir, 'result/best_model/model.npy')
train_loss_path = os.path.join(script_dir, 'result/best_model/trainloss.json')
valid_loss_path = os.path.join(script_dir, 'result/best_model/validlosses.json')
accuracy_path = os.path.join(script_dir, 'result/best_model/val_accuracy.json')
learningrate_path = os.path.join(script_dir, 'result/best_model/learningrates.json')

# 加载数据
with open(train_loss_path, 'r') as f:
    train_loss = json.load(f)
with open(valid_loss_path, 'r') as f:
    valid_loss = json.load(f)
with open(accuracy_path, 'r') as f:
    accuracy = json.load(f)
with open(learningrate_path, 'r') as f:
    learningrate = json.load(f)

tmp1=range(1, len(train_loss)*10 + 1)
tmp1=tmp1[::10]
# 绘制训练损失曲线
plt.figure(figsize=(10, 5))
plt.plot(tmp1, train_loss, label='Training Loss', linewidth=2, color='blue')
plt.xlabel('times')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.savefig(save_path1)  # 保存图片
plt.show()


tmp2=range(1, len(valid_loss)*10 + 1)
tmp2=tmp2[::10]

# 绘制验证损失曲线
plt.figure(figsize=(10, 5))
plt.plot(tmp2, valid_loss, label='Validation Loss', linewidth=2, color='red')
plt.xlabel('times')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(save_path2)  # 保存图片
plt.show()


tmp3=range(1, len(accuracy)*1000 + 1)
tmp3=tmp3[::1000]

# 绘制准确率曲线
plt.figure(figsize=(10, 5))
plt.plot(tmp3, accuracy, label='Validation Accuracy')
plt.xlabel('times')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(save_path3)  # 保存图片
plt.show()

tmp4=range(1, len(train_loss)*10 + 1)
tmp4=tmp1[::10]
# 绘制训练损失曲线
plt.figure(figsize=(10, 5))
plt.plot(tmp1, learningrate, label='learning rate', linewidth=2, color='blue')
plt.xlabel('times')
plt.ylabel('learning rate')
plt.title('learning rate')
plt.legend()
plt.grid(True)
plt.savefig(save_path4)  # 保存图片
plt.show()