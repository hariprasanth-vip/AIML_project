import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import copy
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model definition
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

print(f"Training samples: {len(full_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Split 3 clients
NUM_CLIENTS = 3
total = len(full_dataset)
base = total // NUM_CLIENTS
remainder = total % NUM_CLIENTS
client_sizes = [base] * NUM_CLIENTS
for i in range(remainder):
    client_sizes[i] += 1

client_datasets = random_split(full_dataset, client_sizes)
print(f"Client sizes: {client_sizes}")

# Training
def train_client(model, dataset, epochs=1, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()
    model.train()
    model.to(device)
    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()

# FedAvg
def federated_averaging(client_weights):
    avg_weights = {}
    for key in client_weights[0].keys():
        sum_weights = torch.zeros_like(client_weights[0][key])
        for weights in client_weights:
            sum_weights += weights[key]
        avg_weights[key] = sum_weights / len(client_weights)
    return avg_weights

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
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100 * correct / total

# Federated Learning
global_model = CIFAR10CNN()
global_model.to(device)

NUM_ROUNDS = 10
LOCAL_EPOCHS = 1
round_accuracies = []

print("\nStarting Federated Learning...")
print("="*50)

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\nRound {round_num}/{NUM_ROUNDS}")
    client_weights = []
    
    for client_id, client_data in enumerate(client_datasets):
        local_model = copy.deepcopy(global_model)
        print(f"  Client {client_id+1} training...")
        weights = train_client(local_model, client_data, epochs=LOCAL_EPOCHS)
        client_weights.append(weights)
    
    print(f"  Aggregating weights...")
    avg_weights = federated_averaging(client_weights)
    global_model.load_state_dict(avg_weights)
    
    acc = evaluate(global_model, test_dataset)
    round_accuracies.append(acc)
    print(f"  Accuracy: {acc:.2f}%")

print(f"\nFinal FL Accuracy: {round_accuracies[-1]:.2f}%")