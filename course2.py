import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from tqdm import tqdm

# 1. 配置超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_CLASSES = 100

# 数据预处理
transform_train = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# 加载 CIFAR-100
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# 2. 训练与验证逻辑
def train_one_epoch(model, loader, criterion, optimizer, epoch_idx):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch_idx + 1}")

    for i, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'Loss': f"{running_loss / (i + 1):.4f}",
            'Acc': f"{100. * correct / total:.2f}%"
        })

    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total


# 3. 实验循环
def run_experiment(model_name, use_pretrained):
    exp_name = f"{model_name}_{'FineTune' if use_pretrained else 'Scratch'}"
    print(f"\n>>> 正在运行实验: {exp_name}")

    if model_name == "ResNeXt50":
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if use_pretrained else None
        model = models.resnext50_32x4d(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_name == "DenseNet121":
        weights = models.DenseNet121_Weights.DEFAULT if use_pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    lr = LEARNING_RATE if not use_pretrained else LEARNING_RATE * 0.1
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    start_time = time.time()
    for epoch in range(EPOCHS):
        # 修正处：传入 epoch 参数
        t_loss, t_acc = train_one_epoch(model, trainloader, criterion, optimizer, epoch)
        v_loss, v_acc = validate(model, testloader, criterion)

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['test_acc'].append(v_acc)

        print(f"Epoch {epoch + 1:02d}: Loss={t_loss:.4f}, TrainAcc={t_acc:.2f}%, TestAcc={v_acc:.2f}%")

    duration = (time.time() - start_time) / 60
    print(f"--- {exp_name} 完成! 耗时: {duration:.2f} min ---")
    return history

if __name__ == '__main__':
    results = {}
    configs = [
        ("ResNeXt50", False),
        ("ResNeXt50", True),
        ("DenseNet121", False),
        ("DenseNet121", True)
    ]

    for model_name, is_pre in configs:
        name = f"{model_name}_{'FT' if is_pre else 'Scratch'}"
        results[name] = run_experiment(model_name, is_pre)

    all_dfs = []
    for name, data in results.items():
        temp_df = pd.DataFrame(data)
        temp_df['Model'] = name
        temp_df['Epoch'] = range(1, EPOCHS + 1)
        all_dfs.append(temp_df)

    summary_df = pd.concat(all_dfs)
    summary_df.to_csv("training_results_summary.csv", index=False)
    print("\n✅ 数据已导出至 training_results_summary.csv")

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data['train_loss'], label=name)
    plt.title('Training Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data['test_acc'], label=name)
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("comparison_curves.png", dpi=300)
    plt.show()
    print("✅ 曲线图已保存至 comparison_curves.png")