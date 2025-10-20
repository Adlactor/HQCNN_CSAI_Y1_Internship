import os
import logging
import subprocess

# Suppress TensorFlow logging to only show errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def launch_tensorboard(logdir: str = "runs", port: int = 6070):
    """
    Launch TensorBoard as a subprocess to visualize training logs.

    Parameters:
        logdir (str): Directory where TensorBoard will look for event files.
        port (int): TCP port for TensorBoard to listen on.

    Returns:
        subprocess.Popen: Handle to the launched TensorBoard process.
    """
    # Build the command for launching TensorBoard
    tb_cmd = ["tensorboard", "--logdir", logdir, "--port", str(port), "--reload_interval", "5"]
    # Launch process without blocking, suppressing its output
    return subprocess.Popen(tb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_device():
    """
    Determine whether a GPU (CUDA) is available and return the appropriate torch.device.

    Returns:
        torch.device: 'cuda' if available, else 'cpu'.
    """
    # Use GPU if available for faster computation
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
def get_hyperparameters():
    """
    Define and return common hyperparameters for training.

    Returns:
        dict: Contains 'batch_size', 'learning_rate', and 'epochs'.
    """
    return {
        'batch_size': 64,
        'learning_rate': 0.001,
        'epochs': 10
    }

def get_data_loaders(batch_size):
    """
    Download the MNIST dataset, apply transformations, and wrap in DataLoaders.

    Parameters:
        batch_size (int): Number of samples per batch for both train and test loaders.

    Returns:
        Tuple[DataLoader, DataLoader]: train_loader and test_loader.
    """
    # Define preprocessing pipeline: convert images to tensors and normalize pixel values
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download and prepare the training dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=data_transform)
    # Download and prepare the test dataset
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=data_transform)
    
    # Wrap datasets in DataLoaders for batch iteration
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class CNN(nn.Module):
    """
    Convolutional Neural Network model for MNIST digit classification.

    Architecture:
        - Conv2d(1 -> 32, 3x3, padding=1) + ReLU + MaxPool(2x2)
        - Conv2d(32 -> 64, 3x3, padding=1) + ReLU + MaxPool(2x2)
        - Flatten
        - Linear(64*7*7 -> 128) + ReLU
        - Dropout(0.5)
        - Linear(128 -> 10)
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Define the forward pass through the network.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instantiate model, define loss and optimizer
def initialize_model(learning_rate, device):
    """
    Instantiate the CNN model, loss function, and optimizer.

    Parameters:
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): Device to move the model to ('cpu' or 'cuda').

    Returns:
        Tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
            model: CNN on specified device,
            criterion: CrossEntropyLoss,
            optimizer: Adam optimizer.
    """
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer

# Training the model
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device, writer):
    """
    Train the model and log metrics to TensorBoard.

    Parameters:
        model (nn.Module): The neural network to train.
        train_loader (DataLoader): Loader for training data.
        test_loader (DataLoader): Loader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for weight updates.
        epochs (int): Number of epochs to train.
        device (torch.device): Device for computation.
        writer (SummaryWriter): TensorBoard summary writer.
    """
    global_step = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
            
            # Log per-step scalars
            writer.add_scalar('Step/Train_Loss', loss.item(), global_step)
            batch_acc = (preds == labels).sum().item() / labels.size(0) * 100
            writer.add_scalar('Step/Train_Accuracy', batch_acc, global_step)

            global_step += 1
            
        # Compute epoch-level metrics
        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # Validation loop (no gradient computation)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Validation Epoch {epoch}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        # Log epoch-level scalars
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Test_Loss', val_loss, epoch)
        writer.add_scalar('Epoch/Train_Accuracy', train_acc, epoch)
        writer.add_scalar('Epoch/Test_Accuracy', val_acc, epoch)

        print(f'Epoch {epoch}: ' +
                f'Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, ' +
                f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')

def test_model(model, test_loader, device):
    """
    Evaluate the trained model on the test dataset and print accuracy.

    Parameters:
        model (nn.Module): Trained neural network.
        test_loader (DataLoader): Loader for test data.
        device (torch.device): Device for computation.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

def save_model(model, path='mnist_cnn.pth'):
    """
    Save model state to disk.

    Parameters:
        model (nn.Module): Model to save.
        path (str): File path to save the state dict.
    """
    torch.save(model.state_dict(), path)

# Main function
def main():
    # Launch TensorBoard
    tb_process = launch_tensorboard(logdir='runs/mnist_experiment', port=6070)

    device = get_device()
    params = get_hyperparameters()
    train_loader, test_loader = get_data_loaders(params['batch_size'])
    model, criterion, optimizer = initialize_model(params['learning_rate'], device)
    writer = SummaryWriter(log_dir='runs/mnist_experiment')

    train_model(model, train_loader, test_loader,
                criterion, optimizer,
                params['epochs'], device,
                writer)
    test_model(model, test_loader, device)
    save_model(model)

    writer.close()
    tb_process.terminate()

if __name__ == "__main__":
    main()
