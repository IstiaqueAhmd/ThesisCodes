import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ScalogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.data = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.npy'):
                    self.data.append((os.path.join(class_dir, file), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        scalogram = np.load(file_path).transpose(2, 0, 1)
        scalogram = torch.tensor(scalogram, dtype=torch.float32)
        if self.transform:
            scalogram = self.transform(scalogram)
        return scalogram, label


def calculate_dataset_stats(dataset):
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    mean = torch.zeros(2)
    std = torch.zeros(2)
    for inputs, _ in loader:
        for i in range(2):
            mean[i] += inputs[:, i].mean()
            std[i] += inputs[:, i].std()
    return (mean / len(loader)).tolist(), (std / len(loader)).tolist()


class DualStreamDenseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Create two DenseNet121 streams with modified input channels
        self.stream1 = self._make_densenet_stream()
        self.stream2 = self._make_densenet_stream()

        # Classifier takes concatenated features (1024 * 2 = 2048)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes)
        )

    def _make_densenet_stream(self):
        # Load DenseNet121 without pretrained weights
        model = models.densenet121(pretrained=False)
        # Modify first convolution to accept 1 input channel
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Return feature extractor with pooling
        return nn.Sequential(
            model.features,  # Feature extraction layers
            nn.ReLU(inplace=True),  # Activation after features
            nn.AdaptiveAvgPool2d(1)  # Global pooling
        )

    def forward(self, x):
        # Split input into two modalities
        x1 = self.stream1(x[:, 0:1, :, :])  # First modality [B, 1, H, W]
        x2 = self.stream2(x[:, 1:2, :, :])  # Second modality [B, 1, H, W]

        # Flatten features [B, 1024, 1, 1] -> [B, 1024]
        x1 = x1.flatten(1)
        x2 = x2.flatten(1)

        # Concatenate features from both streams
        x = torch.cat((x1, x2), dim=1)  # [B, 2048]

        # Final classification
        return self.classifier(x)



if __name__ == '__main__':
    # Dataset setup
    train_dir = 'Data/Splitted_Data/train'
    val_dir = 'Data/Splitted_Data/val'
    test_dir = 'Data/Splitted_Data/test'

    # Calculate dataset stats
    temp_train = ScalogramDataset(train_dir)
    mean, std = calculate_dataset_stats(temp_train)
    print(f"Dataset stats - Mean: {mean}, Std: {std}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Normalize(mean, std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])

    # Create datasets
    train_dataset = ScalogramDataset(train_dir, train_transform)
    val_dataset = ScalogramDataset(val_dir, val_test_transform)
    test_dataset = ScalogramDataset(test_dir, val_test_transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = DualStreamDenseNet(len(train_dataset.classes)).to(device)
    print(sum(p.numel() for p in model.parameters()))
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # Store Losses
    train_losses = []
    val_losses = []

    # Training loop with early stopping
    best_val_acc = 0.0
    patience_counter = 0
    patience = 7
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = train_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Calculate metrics
        train_acc = correct / total
        val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)

        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"Learning rate reduced to {new_lr:.6f}")

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f} | Acc: {val_acc:.4f}\n")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "modelv8.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Final evaluation
    model.load_state_dict(torch.load("modelv8.pth"))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

    # Confusion matrices
    plt.figure(figsize=(15, 6))

    # Raw counts
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.title("Confusion Matrix")

    #Loss track
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("confusion_matrices_modelv8.png")
    plt.show()

