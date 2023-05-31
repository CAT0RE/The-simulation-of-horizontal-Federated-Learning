import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import dill
import threading
import random


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 32, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 7 * 7 * 32)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# Load the MNIST test dataset
transform = transforms.Compose([transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Function to train the local model on client data
def train_local_model(client_name, model, train_dataset, num_epochs=5, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'{client_name} Epoch {epoch + 1} Loss: {running_loss / len(train_loader)}')


# Function to perform federated averaging of global model
def federated_average(global_model, local_models, M, random_numbers):
    num_clients = M
    global_state_dict = global_model.state_dict()

    for param_name in global_state_dict:
        avg_param = torch.zeros_like(global_state_dict[param_name])
        for i in random_numbers:
            avg_param += local_models[i].state_dict()[param_name]
        avg_param /= num_clients
        global_state_dict[param_name] = avg_param

    global_model.load_state_dict(global_state_dict)


# Load client training datasets
client_datasets = []
for i in range(1, 21):
    with open(f"C:/Users/123/OneDrive/桌面/FL_Project-1/FL_Project/Client{i}.pkl", 'rb') as f:
        client_dataset = dill.load(f)
        client_datasets.append(client_dataset)

# Create global model and distribute to clients
global_model = CNN()
local_models = [CNN() for _ in range(20)]

global_model.to(device)


def local(client_name, local_models, client_datasets, test_dataset):
    print(f"---Training Local Model {client_name}---")
    train_local_model(client_name, local_models, client_datasets, num_epochs=15, learning_rate=0.001)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images  = images.to(device)
            labels = labels.to(device)
            outputs = local_models(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"<Local_models {client_name} Accuracy on Test Dataset: {accuracy}%>")


# M out of N clients
M = 10


# Stage 2 realization
for round in range(1, 8):
    random_numbers = random.sample(range(20), M)
    random_numbers = sorted(random_numbers)

    print(f"------Current FL round is {round}------")
    weights = global_model.state_dict()
    for client in range(20):
        local_models[client].load_state_dict(weights)
        local_models[client].to(device)

    thread = []

    for i in random_numbers:
        thread.append(threading.Thread(target=local,
                                       args=(f"client-{i+1}", local_models[i], client_datasets[i], test_dataset)))
    for i in range(M):
        thread[i].start()
    for i in range(M):
        thread[i].join()

    # Perform federated averaging to update the global model
    print("---Performing Federated Averaging---")
    federated_average(global_model, local_models, M, random_numbers)

    # Evaluate the global model on the test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = global_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"<Global Model Accuracy on Test Dataset: {accuracy}%>\n")
