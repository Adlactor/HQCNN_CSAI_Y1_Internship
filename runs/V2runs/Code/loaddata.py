from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_mnist_dataloaders(batch_size=32, num_workers=2):
    """
    Create and return DataLoader instances for the MNIST dataset.

    Downloads the MNIST dataset if not already available, applies the necessary
    transformations, and wraps the train and test splits in DataLoaders for efficient
    batch processing with optional multi-process data loading.

    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 2.
        pin_memory (bool, optional): If True, the DataLoader will copy Tensors into CUDA pinned memory
                                        before returning them. Improves GPU transfer performance. Defaults to True.

    Returns:
        tuple: (train_loader, test_loader)
            train_loader (DataLoader): DataLoader for the training set with shuffling enabled.
            test_loader (DataLoader): DataLoader for the test set with shuffling disabled.
    """
    # Define transformations: Convert to tensor and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
    ])

    # Download and create datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
