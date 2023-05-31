import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import socket
import pickle
import threading


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


# Create the global model and distribute to clients
global_model = CNN()

# Define the server address and port
SERVER_ADDRESS = '127.0.0.1'
SERVER_PORT = 12345

# Define the number of clients
NUM_CLIENTS = 20

# Create a TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_ADDRESS, SERVER_PORT))

# Lock for synchronization
lock = threading.Lock()

# List to store received local models
received_local_models = []


def send_client(client_socket, client_address):
    print(f'New connection from {client_address}')
    global_model_params = pickle.dumps(global_model.state_dict())
    client_socket.send(global_model_params)


def receive_client(client_socket, client_address):
    # Receive the local model parameters from the client
    data = client_socket.recv(int(1E6))
    local_model_params = pickle.loads(data)
    local_model = CNN()
    local_model.load_state_dict(local_model_params)
    local_model.to(device)
    received_local_models.append(local_model)


# Load the MNIST test dataset
transform = transforms.Compose([transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

global_model.to(device)


# Function to perform federated averaging of global model
def federated_average(global_model, received_local_models):
    num_clients = len(received_local_models)
    global_state_dict = global_model.state_dict()

    for param_name in global_state_dict:
        avg_param = torch.zeros_like(global_state_dict[param_name])
        for model in received_local_models:
            avg_param += model.state_dict()[param_name]
        avg_param /= num_clients
        global_state_dict[param_name] = avg_param

    global_model.load_state_dict(global_state_dict)


# Start listening for client connections
server_socket.listen(NUM_CLIENTS)
print('Server started and listening for client connections...')

# Socket Communication
for round in range(1, 8):
    print(f"------Current FL round is {round}------")

    # Reset the received_local_models list
    received_local_models = []

    # Accept connections from clients and handle them in separate threads
    threads = []

    for client in range(NUM_CLIENTS):
        client_socket, client_address = server_socket.accept()
        thread = threading.Thread(target=send_client, args=(client_socket, client_address))
        thread.start()
        threads.append(thread)

    # Wait for all client threads to finish
    for thread in threads:
        thread.join()

    for _ in range(NUM_CLIENTS):
        client_socket, client_address = server_socket.accept()
        thread = threading.Thread(target=receive_client, args=(client_socket, client_address))
        thread.start()
        threads.append(thread)

    # Wait for all client threads to finish
    for thread in threads:
        thread.join()

    # Perform federated averaging to update the global model
    federated_average(global_model, received_local_models)

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


# Close the server socket
server_socket.close()
