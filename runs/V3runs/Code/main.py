import os
import logging
import subprocess
import time

# Suppress TensorFlow verbose logs (if any imported modules use TF)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import HQCNN
from loaddata import get_mnist_dataloaders
from tqdm import tqdm
from hdf5_dataset import HDF5Dataset


def launch_tensorboard(logdir: str = "runs", port: int = 6070):
    """
    Launch TensorBoard in a background subprocess.

    Args:
        logdir (str): Directory containing event logs for TensorBoard.
        port (int): Port number for TensorBoard web server.

    Returns:
        subprocess.Popen: Handle to the launched process, to allow termination later.
    """
    tb_cmd = ["tensorboard", "--logdir", logdir, "--port", str(port), "--reload_interval", "5"]
    return subprocess.Popen(tb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def train(model, device, train_loader, optimizer, criterion, epoch, writer):
    """
    Perform one training epoch over the dataset.

    Steps per batch:
        1. Move data and labels to device.
        2. Forward pass to compute outputs.
        3. Compute loss via criterion.
        4. Zero gradients, backpropagate, and apply gradient clipping.
        5. Optimizer step to update weights.
        6. Accumulate loss and accuracy metrics.
        7. Write per-step scalars to TensorBoard.

    Args:
        model (nn.Module): Neural network to train.
        device (torch.device): Compute device (CPU or CUDA).
        train_loader (DataLoader): Iterable over training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        epoch (int): Current epoch number (1-indexed).
        writer (SummaryWriter): TensorBoard writer for logging.

    Returns:
        (epoch_loss, epoch_accuracy): Tuple of average loss and accuracy over epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for step, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)  # data shape: (batch_size, 1, 28, 28)

        # Forward pass
        # Measure forward pass time
        outputs = model(data)  # forward pass

        # Loss computation
        loss = criterion(outputs, target)

        # Backward pass (backpropagation)
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Optimizer step
        optimizer.step()

        running_loss += loss.item() * data.size(0)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        batch_acc = 100.0 * correct / total if total else 0.0

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.2f}%")

        # TensorBoard per‑step logging
        global_step = (epoch - 1) * len(train_loader) + step
        writer.add_scalar('Step/Loss', loss.item(), global_step)
        writer.add_scalar('Step/Accuracy', batch_acc, global_step)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    end_time = time.time()

    print(f'Epoch [{epoch}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {end_time - start_time:.2f}s')
    return epoch_loss, epoch_acc

def evaluate(model, device, test_loader, criterion, epoch, writer):
    """
    Evaluate model performance on the validation/test dataset.

    Args:
        model (nn.Module): Trained neural network.
        device (torch.device): Compute device.
        test_loader (DataLoader): Iterable over test data.
        criterion (nn.Module): Loss function.
        epoch (int): Current epoch number for logging.
        writer (SummaryWriter): TensorBoard writer for logging.

    Returns:
        (avg_loss, avg_accuracy): Tuple of average loss and accuracy.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)  # data shape: (batch_size, 1, 28, 28)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)

            test_loss += loss.item() * data.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = test_loss / len(test_loader.dataset)
    avg_acc = 100 * correct / total

    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.2f}%')

    # TensorBoard epoch‑level logging
    writer.add_scalar('Epoch/Test_Loss', avg_loss, epoch)
    writer.add_scalar('Epoch/Test_Accuracy', avg_acc, epoch)

    return avg_loss, avg_acc

def main():
    """
    Entry point for training and evaluating the HQCNN model.

    Workflow:
        1. Set hyperparameters and device.
        2. Prepare DataLoaders (HDF5-based quantum-encoded states).
        3. Initialize HQCNN model, loss criterion, and optimizer.
        4. Launch TensorBoard for live metrics.
        5. Optionally resume from checkpoint.
        6. Loop over epochs: train and evaluate, logging metrics.
        7. Save checkpoint after each epoch.
        8. Clean up TensorBoard process.
    """
    # Hyperparameters
    patch_size = 2
    stride = 2
    padding = 0
    q = 2               ### Changed to 2 from 4 for V1-3 Checks (Mahmoud Sallam), also modified generate_pairs() @ quantum_circuit.py & self.fc1 @ model.py.
    num_classes = 10
    num_epochs = 10  # Adjust based on computational resources
    batch_size = 64
    learning_rate = 0.001

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # Get DataLoaders
    # train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size, num_workers=2)

    train_ds = HDF5Dataset('eneqr_states.h5', split='train')
    test_ds  = HDF5Dataset('eneqr_states.h5', split='test')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize the model
    model = HQCNN(
        patch_size=patch_size,
        stride=stride,
        padding=padding,
        q=q,
        num_classes=num_classes,
        backend=None,  # Use default simulator
        random_seed=10,
        n_raw_pca_components=784,
        use_precomputed=True
    ).to(device)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    writer = SummaryWriter("runs")  # default directory structure under ./runs/
    tb_proc = launch_tensorboard("runs", port=6040)

    checkpoint_path = 'checkpoint.pth'
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {checkpoint['epoch']}")

    # Training Loop
    for epoch in range(start_epoch, num_epochs +1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch, writer)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion, epoch, writer)

        # TensorBoard epoch‑level logging for training metrics
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Train_Accuracy', train_acc, epoch)

        writer.add_scalar('Epoch/Test_Loss', test_loss, epoch)
        writer.add_scalar('Epoch/Test_Accuracy', test_acc, epoch)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f'Model checkpoint saved at epoch {epoch}')

    writer.close()
    tb_proc.terminate()

if __name__ == '__main__':
    main()
