import argparse
import warnings
import gc
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    DataLoader,
    random_split,
    Subset
)

from torchvision import datasets, transforms
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a cat vs dog classifier"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Size to resize images (square) - larger values provide more detail but take more memory"
    )

    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Enable data augmentation for training (applies transforms to artificially increase dataset size)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training - how many images to process at once"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate - controls how quickly our model learns"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs - each epoch processes entire dataset once"
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer - helps accelarate in relevant directions and dampen oscillations"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Help prevent overfitting by penalizing very large weight in one direction or the other (L2 penalty)"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="cat_dog_classifier.pth",
        help="Path to save the Pytorch model"
    )

    parser.add_argument(
        "--onnx_path",
        type=str,
        default="cat_dog_classifier.onnx",
        help="Path to save ONNX model. A model for model interoperability"
    )

    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation set split ratio - percentage of data used for validation"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Patience for learning rate scheduler - how many epochs to wait before reducing learning rate"
    )

    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping"
    )
    
    parser.add_argument(
        "--early_stopping-patience",
        type=int,
        default=3,
        help="Number of epochs to wait if there is no appreciable improvement"
    )

    parser.add_argument(
        "--early_stopping-min-delta",
        type=float,
        default=0.001,
        help="Minimum change to qualify as improvement"    
    )

    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run inference on a single image instead of training"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the image file for inference"
    )

    parser.add_argument(
        "--model_file",
        type=str,
        default=None,
        help="Path to the model file (.pth or .onnx) for inference"
    )


    return parser.parse_args()


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU(CUDA)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device




def load_data(data_dir, image_size, val_split, batch_size, device, augmentation):
    warnings.filterwarnings("ignore", message="Truncated File Read", category=UserWarning, module= "PIL.TiffImagePlugin")
    use_pin_memory = (device.type != "mps")
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if augmentation:
        print("Using data augmentation during training")
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        print("No data augmentation")
        train_transform = val_transform
    
    try:
        full_dataset = datasets.ImageFolder(root=data_dir)
        print(f"Found {len(full_dataset)} images in total")
        print(f"Classes: {full_dataset.classes}")

        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size

        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size], generator=generator)
        
        
        train_subset = Subset(full_dataset, train_indices.indices)
        train_subset.dataset.transform = train_transform
        
        val_subset = Subset(full_dataset, val_indices.indices)
        val_subset.dataset.transform = val_transform

        print(f"Training set: {len(train_subset)} images")
        print(f"Validation set: {len(val_subset)} images")


    except Exception as e:
        print(f"Error loading data set from {data_dir}: {e}")
        exit(1)


    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=use_pin_memory)

    return train_loader, val_loader

class CatDogCNN(nn.Module):
    def __init__(self, image_size):
        super(CatDogCNN, self).__init__()
        
        # First convolutional block
        # Conv2d layer: Applies 2d convolution to extract visual features
        # 3 input channels (RGB), 32 output feature maps, 3x3 kernel
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # BatchNorm2d layer: normalizes the outputs of the convolutional layer
        # Helps with faster and more stable training
        self.bn1 = nn.BatchNorm2d(32)

        # MaxPool2d layer: Reduces spatial dimensions by taking maximum in 2x2 regions
        # This reduces computation and helps with translation invariance
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Forth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the dimensions for a fully connected layer
        # After 4 pooling layers (each reducing dimensions by half) the size is divded by 16
        feature_size = image_size // 16
        fc_input_size = 256 * feature_size * feature_size

        # Fully connected layers for classification
        # Takes flattened feature maps and outputs class probabilities
        self.fc1 = nn.Linear(fc_input_size, 512)

        # Drop out layer: randomly zeroes some elements during training
        # This prevents overfitting by making the network more robust
        self.dropout = nn.Dropout(0.5)

        # Final layer: outputs tw values for two classes
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()





def run_training_and_cleanup(args, device):
    using_workers = False
    
    try:
        train_loader, val_loader = load_data(
            args.data_dir,
            args.image_size,
            args.val_split,
            args.batch_size,
            device,
            args.augmentation
        )
        using_workers = True
        model = CatDogCNN(args.image_size).to(device)

    except Exception as e:
        print(f"Exception ocurred: {e}")
    finally:
        print("Cleaning up resources")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            gc.collect()
            gc.collect()
        gc.collect()

def main():
    # Parse command line flags
    args = parse_args()

    # Setup device
    device = setup_device()
    
    if args.inference:
        print("Performing inference")
    else:
        print("Training model")
        using_workers = run_training_and_cleanup(args, device)

if __name__ == "__main__":
    main()