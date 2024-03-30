import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.USPS(root='./data', train=True, download=True, transform=transform)
test_data = datasets.USPS(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024, 256)  # Update the input size to 1024
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        print("Size before FC1:", x.size())  # Debug statement
        x = torch.relu(self.fc1(x))
        print("Weight matrix size:", self.fc1.weight.size())  # Debug statement
        x = self.fc2(x)
        return x



def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, conf_matrix

mlp_model = MLP()
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_model(mlp_model, train_loader, criterion, mlp_optimizer)

mlp_accuracy, mlp_precision, mlp_recall, mlp_conf_matrix = evaluate_model(mlp_model, test_loader)

print("MLP Accuracy:", mlp_accuracy)
print("--------")
print("MLP Precision:", mlp_precision)
print("--------")
print("MLP Recall:", mlp_recall)
print("--------")
print("MLP Confusion Matrix:\n", mlp_conf_matrix)
print("--------")

cnn_model = CNN()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

train_model(cnn_model, train_loader, criterion, cnn_optimizer)

cnn_accuracy, cnn_precision, cnn_recall, cnn_conf_matrix = evaluate_model(cnn_model, test_loader)

print("CNN Accuracy:", cnn_accuracy)
print("--------")
print("CNN Precision:", cnn_precision)
print("--------")
print("CNN Recall:", cnn_recall)
print("--------")
print("CNN Confusion Matrix:\n", cnn_conf_matrix)
print("--------")

custom_cnn_model = CustomCNN()
custom_cnn_optimizer = optim.Adam(custom_cnn_model.parameters(), lr=0.001)

train_model(custom_cnn_model, train_loader, criterion, custom_cnn_optimizer)

custom_cnn_accuracy, custom_cnn_precision, custom_cnn_recall, custom_cnn_conf_matrix = evaluate_model(custom_cnn_model, test_loader)

print("Custom CNN Accuracy:", custom_cnn_accuracy)
print("--------")
print("Custom CNN Precision:", custom_cnn_precision)
print("--------")
print("Custom CNN Recall:", custom_cnn_recall)
print("--------")
print("Custom CNN Confusion Matrix:\n", custom_cnn_conf_matrix)
print("--------")
