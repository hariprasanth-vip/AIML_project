import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

print("="*60)
print("🚀 FEDERATED LEARNING FOR PRIVACY-PRESERVING AI")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


print("\nLoading CIFAR-10 Dataset...")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training: {len(full_dataset):,} samples")
print(f"Test: {len(test_dataset):,} samples")


def create_non_iid_clients(dataset, num_clients=5):
    labels = [dataset[i][1] for i in range(len(dataset))]
    label_indices = {i: [] for i in range(10)}
    
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)
    
    client_indices = {i: [] for i in range(num_clients)}
    
    for client_id in range(num_clients):
        digit1 = client_id * 2
        digit2 = digit1 + 1
        
        if digit1 < 10:
            client_indices[client_id].extend(label_indices[digit1][:4000])
        if digit2 < 10:
            client_indices[client_id].extend(label_indices[digit2][:4000])
    
    client_datasets = [torch.utils.data.Subset(dataset, indices) for indices in client_indices.values()]
    return client_datasets

NUM_CLIENTS = 5
print(f"\nCreating {NUM_CLIENTS} Non-IID clients...")

client_datasets = create_non_iid_clients(full_dataset, NUM_CLIENTS)

for i, client in enumerate(client_datasets):
    print(f"   Client {i+1}: {len(client)} samples")


def train_client(model, dataset, epochs=1, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

def evaluate(model, dataset):
    loader = DataLoader(dataset, batch_size=128)
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total

def federated_averaging(client_weights, client_sizes):
    total_samples = sum(client_sizes)
    avg_weights = {}
    
    for key in client_weights[0].keys():
        avg_weights[key] = torch.zeros_like(client_weights[0][key])
        for weights, size in zip(client_weights, client_sizes):
            avg_weights[key] += weights[key] * (size / total_samples)
    
    return avg_weights


print("\n" + "="*60)
print("STARTING FEDERATED LEARNING")
print("="*60)

global_model = CIFAR10CNN().to(device)
client_sizes = [len(client) for client in client_datasets]

NUM_ROUNDS = 10
LOCAL_EPOCHS = 1

fl_accuracies = []

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\nRound {round_num}/{NUM_ROUNDS}")
    client_weights = []
    
    for client_id, client_data in enumerate(client_datasets):
        local_model = copy.deepcopy(global_model)
        weights = train_client(local_model, client_data, epochs=LOCAL_EPOCHS)
        client_weights.append(weights)
        print(f"   Client {client_id+1} done")
    
    avg_weights = federated_averaging(client_weights, client_sizes)
    global_model.load_state_dict(avg_weights)
    
    acc = evaluate(global_model, test_dataset)
    fl_accuracies.append(acc)
    print(f"   ✅ Global Accuracy: {acc:.2f}%")


print("\n" + "="*60)
print("TRAINING CENTRALIZED MODEL")
print("="*60)

central_model = CIFAR10CNN().to(device)
optimizer = torch.optim.Adam(central_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
central_loader = DataLoader(full_dataset, batch_size=128, shuffle=True)

central_accuracies = []

for epoch in range(1, NUM_ROUNDS + 1):
    central_model.train()
    for data, target in central_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = central_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    acc = evaluate(central_model, test_dataset)
    central_accuracies.append(acc)
    print(f"Epoch {epoch}: Accuracy = {acc:.2f}%")


print("\n" + "="*60)
print("PLOTTING RESULTS")
print("="*60)

plt.figure(figsize=(12, 7))

plt.plot(range(1, NUM_ROUNDS + 1), fl_accuracies, 'b-o', linewidth=2, markersize=8, 
         label='Federated Learning', markeredgecolor='black', markeredgewidth=1)
plt.plot(range(1, NUM_ROUNDS + 1), central_accuracies, 'r-s', linewidth=2, markersize=8, 
         label='Centralized Learning', markeredgecolor='black', markeredgewidth=1)

plt.fill_between(range(1, NUM_ROUNDS + 1), fl_accuracies, central_accuracies, 
                  alpha=0.2, color='gray')

plt.xlabel('Rounds (FL) / Epochs (Centralized)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
plt.title('Federated Learning vs Centralized Learning on CIFAR-10', fontsize=16, fontweight='bold')

plt.xticks(range(1, NUM_ROUNDS + 1))
plt.yticks(range(0, 101, 10))
plt.ylim(0, 100)
plt.grid(True, alpha=0.3, linestyle='--')

for i, (fl_acc, cent_acc) in enumerate(zip(fl_accuracies, central_accuracies)):
    plt.annotate(f'{fl_acc:.1f}', (i+1, fl_acc), textcoords="offset points", 
                 xytext=(0, 10), ha='center', fontsize=9, color='blue')
    plt.annotate(f'{cent_acc:.1f}', (i+1, cent_acc), textcoords="offset points", 
                 xytext=(0, -15), ha='center', fontsize=9, color='red')

plt.legend(loc='lower right', fontsize=12, framealpha=0.9)

summary_text = f'FL Final: {fl_accuracies[-1]:.1f}% | Centralized: {central_accuracies[-1]:.1f}%\nGap: {abs(fl_accuracies[-1] - central_accuracies[-1]):.1f}%'
plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('fl_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
