import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集路径
dataset_directory_location = 'Tomato_after'

# 数据预处理和增强
correct_image_shape = (224, 224)
train_transform = transforms.Compose([
    transforms.Resize(correct_image_shape),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(correct_image_shape),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载完整数据集
full_dataset = ImageFolder(dataset_directory_location)

# 创建训练集、验证集和测试集的索引
# 首先将数据分成训练集(80%)和临时集(20%)
train_idx, temp_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)
# 然后将临时集平均分成验证集(10%)和测试集(10%)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

# 创建训练集、验证集和测试集的采样器
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetRandomSampler(test_idx)

# 创建数据集
train_dataset = ImageFolder(dataset_directory_location, transform=train_transform)
val_dataset = ImageFolder(dataset_directory_location, transform=val_test_transform)
test_dataset = ImageFolder(dataset_directory_location, transform=val_test_transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=16, sampler=val_sampler)
test_loader = DataLoader(test_dataset, batch_size=16, sampler=test_sampler)

# 打印数据集大小
print(f"Total dataset size: {len(full_dataset)}")
print(f"Training set size: {len(train_idx)} ({len(train_idx) / len(full_dataset) * 100:.1f}%)")
print(f"Validation set size: {len(val_idx)} ({len(val_idx) / len(full_dataset) * 100:.1f}%)")
print(f"Test set size: {len(test_idx)} ({len(test_idx) / len(full_dataset) * 100:.1f}%)")


class TomatoDiseaseModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TomatoDiseaseModel, self).__init__()
        # 创建ResNet50但不使用预训练权重
        self.resnet = models.resnet50(pretrained=False)

        # 修改最后的分类层以匹配类别数
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


# 初始化模型
model = TomatoDiseaseModel().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# 由于是从头训练，所有层都使用相同的学习率
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 评估函数
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average='weighted'
    )

    # 计算准确率
    accuracy = sum(1 for x, y in zip(all_predictions, all_labels) if x == y) / len(all_labels)

    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }


# 训练函数
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算训练指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average='weighted'
    )

    accuracy = sum(1 for x, y in zip(all_predictions, all_labels) if x == y) / len(all_labels)

    return {
        'loss': running_loss / len(train_loader),
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }


# 训练模型
epochs = 50
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': [],
    'train_precision': [], 'val_precision': [],
    'train_recall': [], 'val_recall': [],
    'train_f1': [], 'val_f1': []
}

for epoch in range(epochs):
    # 训练阶段
    train_metrics = train_epoch(model, train_loader, criterion, optimizer)

    # 验证阶段
    val_metrics = evaluate(model, val_loader, criterion)

    # 测试阶段
    test_metrics = evaluate(model, test_loader, criterion)

    # 记录历史
    history['train_loss'].append(train_metrics['loss'])
    history['val_loss'].append(val_metrics['loss'])
    history['train_acc'].append(train_metrics['accuracy'])
    history['val_acc'].append(val_metrics['accuracy'])
    history['train_precision'].append(train_metrics['precision'])
    history['val_precision'].append(val_metrics['precision'])
    history['train_recall'].append(train_metrics['recall'])
    history['val_recall'].append(val_metrics['recall'])
    history['train_f1'].append(train_metrics['f1'])
    history['val_f1'].append(val_metrics['f1'])

    print(f'Epoch {epoch + 1}/{epochs}:')
    print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%, "
          f"P: {train_metrics['precision']:.4f}%, R: {train_metrics['recall']:.2f}%, F1: {train_metrics['f1']:.2f}%")
    print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, "
          f"P: {val_metrics['precision']:.4f}%, R: {val_metrics['recall']:.2f}%, F1: {val_metrics['f1']:.2f}%")
    print(f"Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.2f}%, "
          f"P: {test_metrics['precision']:.4f}%, R: {test_metrics['recall']:.2f}%, F1: {test_metrics['f1']:.2f}%")
    print('-' * 100)

# 在测试集上进行最终评估
test_metrics = evaluate(model, test_loader, criterion)
print("\n测试集评估结果:")
print(f"准确率: {test_metrics['accuracy']:.2f}%")
print(f"F1分数: {test_metrics['f1']:.2f}%")
print(f"精确率: {test_metrics['precision']:.2f}%")
print(f"召回率: {test_metrics['recall']:.2f}%")

# 绘制训练历史
plt.figure(figsize=(15, 10))

# 损失曲线
plt.subplot(2, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 准确率曲线
plt.subplot(2, 2, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# F1分数曲线
plt.subplot(2, 2, 3)
plt.plot(history['train_f1'], label='Train')
plt.plot(history['val_f1'], label='Validation')
plt.title('F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score (%)')
plt.legend()

# 召回率曲线
plt.subplot(2, 2, 4)
plt.plot(history['train_recall'], label='Train')
plt.plot(history['val_recall'], label='Validation')
plt.title('Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall (%)')
plt.legend()

plt.tight_layout()
plt.show()


# 计算并绘制混淆矩阵
def plot_confusion_matrix(model, test_loader, class_names):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 获取类别名称并绘制混淆矩阵
class_names = full_dataset.classes
plot_confusion_matrix(model, test_loader, class_names)