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


    def forward(self, x):
        """
        Forward pass through the network.
        
        This defines how the input flows through the layers to produce an output.
        Each operation transforms the tensor shape and content progressively

        Args: 
            x (Tensor): Input tensor of shape [batch_size, 3, image_size, image_size]

        Returns:
            Tensor: Output tensor of shape [batch_size, 2]
                    Contains logits for each class (cat=0, dog=1)
        """
        # Convolutional feature extraction
        for i in range(1, 5):
            # Dynamically get the layer for the block(conv1, bn1, pool1, etc.)
            # getattr(self, f'conv{i}')
            conv = getattr(self, f'conv{i}')
            bn = getattr(self, f'bn{i}')
            pool = getattr(self, f'pool{i}')

            # Step 1: Apply 2d convolution
            x = conv(x)

            # Step 2: Apply batch normalization
            x = bn(x)

            # Step 3: Apply ReLU activation
            x = self.relu(x)

            # Step 4: Aplly max pooling
            x = pool(x)

        # After all 4 blocks, tensor shape is approc. [batch, 256, image_size /16]

        # Prepate for classification
        # Flatten 4D feature maps into 2d for fully connected layer
        x = x.view(x.size(0), -1)

        # Classification layers
        # First fully connected layer
        x = self.fc1(x)

        # Apply ReLU activation
        x = self.relu(x)

        # Drop out layer
        x = self.dropout(x)

        # Final classification layer
        x = self.fc2(x)

        return x





class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"Early stopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggerd!")
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop




def train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs,
        early_stopping_enabled=False,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001
):
    """
    Train the model and validate it after each epoch.

    This function:
    1. Trains the model on the training data
    2. Validates the model on the validation data
    3. Adjusts the learning rate if needed
    4. Implements early stopping if enabled
    5. Tracks and returns the best model state

    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): Data loader for training data
        val_loader (DataLoader): Data loader for validation data,
        criterion (nn.Module): Loss function (e.g. CrossEntropyLoss),
        optimizer (optim.Optimizer): Optimization algorithm (e.g. SGD),
        scheduler (optim.lr_scheduler): Learning rate scheduler,
        device (torch.device): Device to use for computation,
        num_epochs (int): Maximum number of training epochs,
        early_stopping_enabled (bool): Whether to use early stopping,
        early_stopping_patience (int): Number of epochs to wait before stopping,
        early_stopping_min_delta (float): Minimum change to qualify as improvement
    
    Returns:
        tuple: (best_model_state, best_val_accuracy) Best model state and its validation accuracy
    """

    print("\nStarting training...")
    training_loss = []
    val_accuracies = []

    best_val_accuracy = 0.0
    best_model_state = None

    early_stopper = None
    if early_stopping_enabled:
        early_stopper = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)
        print(f"Early stopping enabled with patience = {early_stopping_patience}")


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")
        
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss
            train_loader_tqdm.set_postfix(loss=running_loss / (train_loader_tqdm.n + 1))

        avg_train_loss = running_loss / len(train_loader)
        training_loss.append(avg_train_loss)

        # Validation phase
        model.eval()
        correct_predictions = 0
        total_samples = 0
        val_running_loss = 0.0

        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)")
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                val_loader_tqdm.set_postfix(
                    accuracy=f"{100 * correct_predictions / total_samples:.2f}%"
                )

        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * correct_predictions / total_samples
        val_accuracies.append(val_accuracy)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current learning rate: {current_lr}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()

            print(f"New best model: {best_val_accuracy:.2f}% accuracy")

        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Train loss={avg_train_loss:.2f}, "
            f"Val loss={avg_val_loss:.4f}, "
            f"Val accuracy={val_accuracy:.2f}%"
        )

        if early_stopping_enabled and early_stopper(avg_val_loss):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print(f"Training finished! BEst Validation accuracy: {best_val_accuracy:.2f}%")
    
    return best_model_state, best_val_accuracy



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
        print("Data loading completed successfully")
        using_workers = True
        
        print("Initializing neural network model..")
        print("Model architecture: ")
        print(model)
        model = CatDogCNN(args.image_size).to(device)

        # Calculate and display total number of trainable parameters
        # This gives us inside into model complexity and memory reqirements
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")

        # Configure loss function
        print("Setting up training components...")
        criterion = nn.CrossEntropyLoss()

        # Configure optimizer
        optimizer = optim.SGD(
            model.parameters(), # All model weights and biases to optimize
            lr=args.learning_rate, # Step size for weight updates
            momentum=args.momentum, # Momentum factor for smoother convergernce
            weight_decay=args.weight_decay, # Regularization
        )

        # Configure learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.1,
            patience=args.patience
        )

        # Training loop execution
        
        print("Starting the training process...")
        print(f"Training for maximum {args.num_epochs} epochs...")
        if args.early_stopping:
            print(f"Early stopping enabled. Will stop if no improvement for {args.early_stopping_patience} epochs")
        best_model_state, best_validation_accuracy = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            args.num_epochs,
            args.early_stopping,
            args.early_stopping_patience,
            args.early_stopping_min_delta
        )

        # Best model restoration
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Restore best model state (validation accuracy: {best_validation_accuracy:.2f})")
        else:
            print("Warning: No best model state saved, using final epoch model.")

        # Model Persistence
        print("Saving trained model...")
        save_model(model, args.model_path, args.onnx_path, args.image_size, device)
        print("Model saved successfully!")
        print("\n" + "=" *50)
        print("ALL TRAINING OPERATIONS COMPLETED SUCCESSFULLY")
        print("\n" + "=" *50)

        # Immediate cleanup of large objects for large operations
        del train_loader
        del val_loader
        del model
        return using_workers
    
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Proceeding with cleanup and resource deallocation")
        return using_workers
    finally:
        print("Cleaning up resources")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            gc.collect()
            gc.collect()
        gc.collect()


def save_model(model, model_path, onnx_path, image_size, device):
    torch.save(model.state_dict(),model_path)
    print(f"Pytoch model saved to {model_path}")

    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"},
                              "output": {0: "batch_size"}}
                )
        print(f"Model exported to ONNX format at {onnx_path}")

    except Exception as e:
        print(f"Error during ONNX export: {e}")




def run_inference(image_path, model_file, image_size, device):
    import numpy as np
    from PIL import Image

    if not Path(image_path).exists() or not Path(model_file).exists():
        print("Error image or model not found")
        return None, None
    
    transform = transforms.Compose([
        transforms.Resize(image_size, image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None
    
    if Path(model_file).suffix == ".pth":
        model = CatDogCNN(image_size).to(device)
        try:
            model.load_state_dict(torch.load
                                  (model_file, map_location=device))
            model.eval()
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                pred_class = ["cat", "dog"][predicted.item()]
                conf_score = confidence.item()
    
        except Exception as e:
            print(f"Error with PyTorch file: {e}")
            return None, None

    elif Path(model_file).suffix == ".onnx":
        try:
            import onnxruntime as ort
            
            ort_session = ort.InferenceSession(model_file)
            ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            outputs = ort_outputs[0]

            def softmax(x):
                exp_x = np.exp(x - np.max(x))
                return exp_x / exp_x.sum()
            
            probabilities = softmax(outputs[0])
            predicted = np.argmax(probabilities)
            pred_class = ["cat", "dog"][predicted]
            conf_score = float(probabilities[predicted])

        except Exception as e:
            print(f"Error with ONNX file: {e}")
            return None, None

    else:
        print("Error: unsupported model format")
        return None, None

    print(f"\nInference results: \nImage: {image_path}\nPrediction: {pred_class}\nConfidence: {conf_score:.2f}")
    return pred_class, conf_score

def main():
    # Parse command line flags
    args = parse_args()

    # Setup device
    device = setup_device()
    
    if args.inference:
        print("Performing inference")
        if not args.image_path:
            print("Error:--image_path is required for inference")
            exit(1)

        if args.model_file:
            model_file = args.model_file
        elif Path(args.path.model_path).exists():
            model_file = args.model_path
            print(f"Using default PyTorch model: {model_file}")
        elif Path(args.onnx_path).exists():
            model_file = args.onnx_path
            print(f"Using default ONNX model: {model_file}")
        else:
            print("Error: No trained model found")
            exit(1)

        run_inference(args.image_path, model_file, args.image_size, device)
        return
    else:
        print("Training model")
        using_workers = run_training_and_cleanup(args, device)
        if using_workers:
            print("Forcing clean exit...")
            os._exit(0)

if __name__ == "__main__":
    main()