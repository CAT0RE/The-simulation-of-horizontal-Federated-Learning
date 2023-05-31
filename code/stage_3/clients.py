import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import dill
import threading
import socket
import pickle

# Define the server address and port
SERVER_ADDRESS = '127.0.0.1'
SERVER_PORT = 12345


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


local_models = [CNN() for _ in range(20)]
lock = threading.Lock()


# Function to send local model parameters to the server and receive global model parameters
def send_model_parameters(client):
    # Connect to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_ADDRESS, SERVER_PORT))

    # Send the local model parameters to the server
    local_model_params = pickle.dumps(local_models[client].state_dict())
    client_socket.send(local_model_params)

    # # Close the client socket
    client_socket.close()


def receive_model_parameters(client):
    # Connect to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_ADDRESS, SERVER_PORT))

    # Receive the global model parameters from the server
    data = client_socket.recv(int(1E6))
    global_model_params = pickle.loads(data)

    # Update the local model with the received global model parameters
    local_models[client].load_state_dict(global_model_params)

    # Close the client socket
    client_socket.close()


# Load client training datasets
client_datasets = []
for i in range(1, 21):
    with open(f"C:/Users/123/OneDrive/桌面/FL_Project-1/FL_Project/Client{i}.pkl", 'rb') as f:
        client_dataset = dill.load(f)
        client_datasets.append(client_dataset)

# Load the MNIST test dataset
transform = transforms.Compose([transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in range(20):
    local_models[i].to(device)


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


def local(client):
    receive_model_parameters(client)

    train_local_model(f"Client-{client + 1}", local_models[client], client_datasets[client],
                      num_epochs=15, learning_rate=0.001)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = local_models[client](images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"<Local_models {client} Accuracy on Test Dataset: {accuracy}%>")

    send_model_parameters(client)


for round in range(7):
    thread = []

    for client in range(20):
        thread.append(threading.Thread(target=local, args=(client,)))
    for i in range(20):
        thread[i].start()
    for i in range(20):
        thread[i].join()

