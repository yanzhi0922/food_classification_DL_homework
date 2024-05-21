import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
import pillow_avif
from PIL import Image
import matplotlib.pyplot as plt

# 读取图像
def readfile(path, needlabel):
    # 初始化图像和标签的数组
    images = []
    labels = []
    # 递归遍历所有文件夹和文件,root表示当前正在访问的文件夹路径,dirs表示该文件夹下的子目录,files表示该文件夹下的文件
    for root, dirs, _ in os.walk(path):
        # print("root: ", root)
        # print("dirs: ", dirs)
        for dir in dirs:
            # print("dir: ", dir)
            file_path = os.path.join(root, dir)
            # print("file_path: ", file_path)
            for file in os.listdir(file_path):
                # print("file: ", file)
                # print(os.path.join(file_path, file))
                img = Image.open(os.path.join(file_path, file))
                if img.mode != 'RGB':
                    img = img.convert('RGB')  # 确保图像是 RGB 格式
                img_resized = img.resize((224, 224))
                # 将图像转为 numpy array
                img_resized = np.array(img_resized)
                images.append(img_resized)
                if needlabel:
                    # 获取最里层文件夹的名称作为标签，映射到数字
                    label = dir
                    if label == "Bread":
                        labels.append(0)
                    elif label == "Hamburger":
                        labels.append(1)
                    elif label == "Kebab":
                        labels.append(2)
                    elif label == "Noodle":
                        labels.append(3)
                    elif label == "Rice":
                        labels.append(4)
    # 将图像列表转换为numpy数组
    x = np.array(images)
    if needlabel:
        y = torch.LongTensor(labels)
        # print("y: ", y)
        return x, y
    else:
        return x

workspace_dir = './Food_Classification_Dataset'
print("Reading data")
# print(os.path.join(workspace_dir, "train"))
train_x, train_y = readfile(os.path.join(workspace_dir, "train"), True)
# print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "val"), True)
# print("Size of validation data = {}".format(len(val_x)))
test_x, test_y = readfile(os.path.join(workspace_dir, "test"), True)
# print("Size of Testing data = {}".format(len(test_x)))
print("Reading data complicated\n")



# 数据增强、数据处理
print("Dataset")
# training 时做 data augmentation
# transforms.Compose 将图像操作串联起来
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 随机将图片水平翻转
    transforms.RandomRotation(15),  # 随机旋转图片 (-15,15)
    transforms.ToTensor(),  # 将图片转成 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# testing 时不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        if y is not None:
            self.y = y
        else:
            self.y = None
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:  # 如果没有标签那么只返回X
            return X


batch_size = 32
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
test_set = ImgDataset(test_x, test_y, test_transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
print("Dataset complicated\n")


# 定义模型
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # 使用预训练的ResNet18模型
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        # 修改最后的 fully connected layer 使得输出维度为 5
        self.resnet.fc = nn.Linear(512, 5)

    def forward(self, x):
        x = self.resnet(x)
        return x

model = Classifier().cuda()
loss = nn.CrossEntropyLoss()  # 因为是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer 使用 Adam
num_epoch = 50
print("Model complicated\n")

# 开始训练
print("Training")
best_val_acc = 0.0
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # 确保 model 是在 train model (开启 Dropout 等...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # 用 optimizer 将 model 参数的 gradient 归零
        train_pred = model(data[0].cuda())  # 利用 model 得到预测的概率分布
        batch_loss = loss(train_pred, data[1].cuda())  # 计算 loss（注意 prediction 跟 label 必须同时在 CPU 或是 GPU 上）
        batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新参数值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        train_accuracies.append(train_acc / train_set.__len__())
        train_losses.append(train_loss / train_set.__len__())
        val_accuracies.append(val_acc / val_set.__len__())
        val_losses.append(val_loss / val_set.__len__())

        # 将结果 print 出来
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val  Acc: %3.6f loss: %3.6f' % (
            epoch + 1, num_epoch, time.time() - epoch_start_time, train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(), val_loss / val_set.__len__()))

        if val_acc / val_set.__len__() > best_val_acc:
            best_val_acc = val_acc / val_set.__len__()
            torch.save(model, "model_resnet18.pth")

# 保存模型, 以便之后测试
# torch.save(model, "model_resnet18.pth")
print("Training complicated\n")

# 开始测试
print("Testing")
model = torch.load("model_resnet18.pth")  # 加载模型
model.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        images, labels = data  # 解包数据
        images = images.cuda()  # 将图像张量移动到 GPU 上
        labels = labels.cuda()  # 将标签张量移动到 GPU 上

        # 现在调用模型进行预测
        test_pred = model(images)  # 注意这里不再调用 .cuda() 因为 model 里已经包含了 .cuda()

        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

# 输出正确率
print("Accuracy: ", sum(prediction == test_y.numpy()) / len(test_y.numpy()))

# 将结果写入 csv 档
with open("predict_resnet18.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(prediction):
        if(y==0):
            f.write('{},{}\n'.format(i, "Bread"))
        elif(y==1):
            f.write('{},{}\n'.format(i, "Hamburger"))
        elif(y==2):
            f.write('{},{}\n'.format(i, "Kebab"))
        elif(y==3):
            f.write('{},{}\n'.format(i, "Noodle"))
        elif(y==4):
            f.write('{},{}\n'.format(i, "Rice"))
print("Testing complicated\n")

# 可视化
plt.figure()
plt.plot(train_accuracies, label='train accuracy')
plt.plot(val_accuracies, label='validation accuracy')
plt.scatter(np.argmax(val_accuracies), max(val_accuracies), c='r', label='best validation accuracy')
plt.scatter(np.argmax(val_accuracies), sum(prediction == test_y.numpy()) / len(test_y.numpy()), c='g', label='test accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()