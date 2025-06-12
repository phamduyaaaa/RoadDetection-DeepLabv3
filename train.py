import os
import time
import random
from typing import Tuple 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# NEW: Import matplotlib để vẽ biểu đồ
import matplotlib.pyplot as plt

from dataset import RoadDataset
from model import DeepLabv3
from metrics import mean_IoU, pixel_accuracy
from utils import save_images


def train(model, optimizer, criterion, n_epoch, data_loaders: dict, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    train_losses = np.zeros(n_epoch)
    val_losses = np.zeros(n_epoch)
    train_accuracies = np.zeros(n_epoch)
    val_accuracies = np.zeros(n_epoch)
    train_ious = np.zeros(n_epoch)
    val_ious = np.zeros(n_epoch)
    
    best_iou = 0.0

    model.to(device)

    since = time.time()

    for epoch in range(n_epoch):
        current_train_loss = 0.0
        current_train_accuracy = 0.0
        current_train_iou = 0.0

        model.train()
        for inputs, targets in tqdm(data_loaders['train'], desc=f'Training... Epoch: {epoch + 1}/{n_epoch}'):
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            current_train_loss += loss.item()
            current_train_accuracy += pixel_accuracy(outputs, targets)
            current_train_iou += mean_IoU(outputs, targets)

            loss.backward()
            optimizer.step()

        train_losses[epoch] = current_train_loss / len(data_loaders['train'])
        train_accuracies[epoch] = current_train_accuracy / len(data_loaders['train'])
        train_ious[epoch] = current_train_iou / len(data_loaders['train'])

        with torch.no_grad():
            current_val_loss = 0.0
            current_val_accuracy = 0.0
            current_val_iou = 0.0
            model.eval()
            for inputs, targets in tqdm(data_loaders['validation'], desc=f'Validating... Epoch: {epoch + 1}/{n_epoch}'):
                inputs, targets = inputs.to(device).float(), targets.to(device).float()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                current_val_loss += loss.item()
                current_val_accuracy += pixel_accuracy(outputs, targets)
                current_val_iou += mean_IoU(outputs, targets)

            val_losses[epoch] = current_val_loss / len(data_loaders['validation'])
            val_accuracies[epoch] = current_val_accuracy / len(data_loaders['validation'])
            val_ious[epoch] = current_val_iou / len(data_loaders['validation'])
        
        if val_ious[epoch] > best_iou: 
            best_iou = val_ious[epoch]
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "epoch": epoch,
                "loss": val_losses[epoch],
                "pixel_accuracy": val_accuracies[epoch],
                "iou": val_ious[epoch]
            }, SAVE_MODEL_DIR)

        print(f"Epoch [{epoch+1}/{n_epoch}]:")
        print(f"Train Loss: {train_losses[epoch]:.4f}, Train Pixel Accuracy: {train_accuracies[epoch]:.4f}, Train IOU: {train_ious[epoch]:.4f}")
        print(f"Validation Loss: {val_losses[epoch]:.4f}, Validation Pixel Accuracy: {val_accuracies[epoch]:.4f}, Validation IOU: {val_ious[epoch]:.4f}")
        print('-'*20)

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return train_losses, val_losses, train_accuracies, val_accuracies, train_ious, val_ious


if __name__ == '__main__':
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    INPUT_SIZE = 256
    BATCH_SIZE = 16
    EPOCHS = 40

    DATASET_DIR_ROOT = 'dataset'
    SAVE_MODEL_DIR = 'weight/model.pt'
    OUTPUT_DIR = 'output_images'
    
    PLOT_DIR = 'plot' 
    
    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda:0'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'

    transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), 2),
        transforms.ToTensor()
    ])

    
    train_dataset = RoadDataset(os.path.join(DATASET_DIR_ROOT, 'Train', 'image'),
                                os.path.join(DATASET_DIR_ROOT, 'Train', 'label'),
                                transforms)
    
    validation_dataset = RoadDataset(os.path.join(DATASET_DIR_ROOT, 'Validation', 'image'),
                                     os.path.join(DATASET_DIR_ROOT, 'Validation', 'label'),
                                     transforms)
    

    data_loaders = {
        'train': DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
        ),
        'validation': DataLoader(
            validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
        )
    }

    
    model = DeepLabv3()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses_hist, val_losses_hist, \
    train_accuracies_hist, val_accuracies_hist, \
    train_ious_hist, val_ious_hist = train(model, optimizer, criterion, EPOCHS, data_loaders, DEVICE)

    os.makedirs(PLOT_DIR, exist_ok=True) 

    np.save(os.path.join(PLOT_DIR, 'train_losses.npy'), train_losses_hist)
    np.save(os.path.join(PLOT_DIR, 'val_losses.npy'), val_losses_hist)
    np.save(os.path.join(PLOT_DIR, 'train_accuracies.npy'), train_accuracies_hist)
    np.save(os.path.join(PLOT_DIR, 'val_accuracies.npy'), val_accuracies_hist)
    np.save(os.path.join(PLOT_DIR, 'train_ious.npy'), train_ious_hist)
    np.save(os.path.join(PLOT_DIR, 'val_ious.npy'), val_ious_hist)

    print(f"Lịch sử huấn luyện đã được lưu vào thư mục '{PLOT_DIR}'")

    checkpoint = torch.load(SAVE_MODEL_DIR, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    save_images(model, data_loaders['validation'], OUTPUT_DIR, DEVICE)

    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses_hist, label='Train Loss')
    plt.plot(epochs_range, val_losses_hist, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accuracies_hist, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies_hist, label='Validation Accuracy')
    plt.title('Pixel Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_ious_hist, label='Train IoU')
    plt.plot(epochs_range, val_ious_hist, label='Validation IoU')
    plt.title('Mean IoU over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # NEW: Lưu biểu đồ tổng hợp vào thư mục PLOT_DIR
    plt.savefig(os.path.join(PLOT_DIR, 'training_metrics_plot.png')) 
    plt.show() # Hiển thị biểu đồ (sẽ mở cửa sổ)
