import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
learning_rate = 0.001
epochs = 15

# 2. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# 3. 模型定义
class OptimalMLP(nn.Module):
    def __init__(self):
        super(OptimalMLP, self).__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.main(x)


model = OptimalMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. 训练过程
loss_history = []
print("Training started...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

# 5. 生成图表 A: Loss 曲线
plt.figure(figsize=(6, 4))
plt.plot(loss_history, 'b-o', label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.title('Training Convergence')
plt.grid(True)
plt.legend()
plt.savefig('loss_curve.png', dpi=300)
plt.close()

# 6. 生成图表 B: 分类示例
model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)
classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
outputs = model(images)
_, preds = torch.max(outputs, 1)

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    img = images[i].cpu().numpy().squeeze()
    plt.imshow(img, cmap='gray')
    color = 'green' if preds[i] == labels[i] else 'red'
    plt.title(f"P: {classes[preds[i]]}\nT: {classes[labels[i]]}", color=color, fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.savefig('results_demo.png', dpi=300)
print("Finished. Images saved as loss_curve.png and results_demo.png")