
# 食物分类深度学习模型

欢迎来到本项目的`README.md`文件！在这里，您将找到如何使用我们提供的代码来复现食物分类深度学习模型实验的详细步骤。

## 环境配置

在开始之前，请确保您的计算环境中安装了以下软件和库：

- Python 3.8 或更高版本
- PyTorch 1.7 或更高版本
- torchvision
- pillow-avif (用于读取 AVIF 格式图片)
- numpy
- matplotlib (用于结果可视化)

您可以通过以下命令安装所需的 Python 库：

```bash
pip install torch torchvision numpy pillow-avif matplotlib
```

## 数据集准备

请将您的数据集放置在`Food_Classification_Dataset`文件夹中，确保数据集的文件夹结构如下：

```
Food_Classification_Dataset/
│
├── train/          # 训练集
│   ├── Bread/     # 面包类别的图像
│   ├── Hamburger/
│   ├── Kebab/
│   ├── Noodle/
│   └── Rice/
│
├── val/           # 验证集
│   ├── Bread/
│   ├── Hamburger/
│   ├── Kebab/
│   ├── Noodle/
│   └── Rice/
│
└── test/          # 测试集
    ├── Bread/
    ├── Hamburger/
    ├── Kebab/
    ├── Noodle/
    └── Rice/
```

## 代码使用

4个模型分别对应四个不同的代码food_classification.py(自定义的)、mobilenet.py、densenet.py、resnet18.py,模型参数分别保存在model.pth、model_densenet.pth、model_mobilenet.pth、model_resnet18.pth

## 实验结果

实验结果，包括模型参数、分类结果csv、记录的txt、Introduction.docx等。

## 注意事项

- 确保图像文件名不要包含特殊字符，以免读取错误。
- 如果您的数据集结构与上述不同，请修改`food_classification.py`中的`readfile`函数以匹配您的文件夹结构。
- 如果您使用的是 AVIF 格式的图像，请确保已经安装了`pillow-avif`库。

## 许可和版权

本项目遵循 [MIT 许可证](LICENSE)。请在遵守许可协议的前提下使用本项目的代码。

