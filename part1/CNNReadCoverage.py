import os
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from DataProcess import load_coverage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')


class CoverageDataset(Dataset):
    def __init__(self, coverage_dict, window_size=1000):
        self.window_size = window_size
        half = window_size // 2
        all_windows = []
        all_targets = []
        for srx_id, chrom_dict in coverage_dict.items():
            for chrom, coverage in chrom_dict.items():
                n = len(coverage)
                if n < window_size:
                    continue
                shape   = (n - window_size + 1, window_size)
                strides = (coverage.strides[0], coverage.strides[0])
                windows = np.lib.stride_tricks.as_strided(coverage, shape=shape, strides=strides)
                targets = coverage[half : half + len(windows)]
                all_windows.append(windows.copy())
                all_targets.append(targets.copy())

        self.windows = np.concatenate(all_windows, axis=0).astype(np.float32)
        self.targets = np.concatenate(all_targets, axis=0).astype(np.float32)
        print(f"Dataset: {len(self.windows)} windows of size {window_size}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32).unsqueeze(0)  # (1, window_size)
        y = torch.tensor([self.targets[idx]], dtype=torch.float32)
        return x, y


def get_dataloaders(train_val_coverage, test_coverage, window_size=500,
                    batch_size=1024, val_split=0.2):
    print("Building train/val dataset...")
    train_val_dataset = CoverageDataset(train_val_coverage, window_size=window_size)

    print("Building test dataset...")
    test_dataset = CoverageDataset(test_coverage, window_size=window_size)

    total      = len(train_val_dataset)
    val_size   = int(total * val_split)
    train_size = total - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train: {train_size} | Val: {val_size} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


class CNNMultipleLayers(nn.Module):
    def __init__(self, num_kernels=8, kernel_size=12, window_size=1000):
        super(CNNMultipleLayers, self).__init__()
        input_channels = 1  # single coverage track

        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, num_kernels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(num_kernels, num_kernels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(num_kernels, num_kernels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.regression_block = nn.Sequential(
            nn.Linear(num_kernels, num_kernels),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(num_kernels, 1),
            nn.Softplus()
        )

    def forward(self, x):
        # x shape: (batch, 1, window_size)
        x = self.conv_block(x)
        x, _ = torch.max(x, dim=2)
        x = self.regression_block(x)
        return x


# --- Poisson Loss ---
def poisson_loss(y_pred, y_true):
    eps = 1e-8
    return torch.mean(y_pred - y_true * torch.log(y_pred + eps))


# --- Training ---
def train_epoch(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    num_batches = len(dataloader)
    total_loss  = 0

    for X, y in dataloader:
        X, y   = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss   = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    if epoch % 10 == 0:
        print(f"Train loss: {avg_loss:>7f}")
    return avg_loss


# --- Validation ---
def validation(dataloader, model, loss_fn, epoch):
    model.eval()
    num_batches = len(dataloader)
    total_loss  = 0
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y   = X.to(device), y.to(device)
            y_pred = model(X)
            total_loss += loss_fn(y_pred, y).item()
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss    = total_loss / num_batches
    all_preds   = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    pearson_r   = np.corrcoef(all_preds, all_targets)[0, 1]

    if epoch % 10 == 0:
        print(f"Val loss: {avg_loss:>8f} | Pearson r: {pearson_r:>8f}")
    return avg_loss


def evaluate(model, dataloader, loss_fn=poisson_loss, name='test'):
    """Evaluate model using Poisson loss and Pearson correlation."""
    model.eval()
    all_preds   = []
    all_targets = []
    num_batches = len(dataloader)
    total_loss  = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y   = X.to(device), y.to(device)
            y_pred = model(X)
            total_loss += loss_fn(y_pred, y).item()
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss    = total_loss / num_batches
    all_preds   = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    pearson_r   = np.corrcoef(all_preds, all_targets)[0, 1]

    print(f"\n--- {name} Evaluation ---")
    print(f"Poisson loss : {avg_loss:>8f}")
    print(f"Pearson r    : {pearson_r:>8f}")
    return avg_loss, pearson_r


def save_modelCNN(model, train_losses, val_losses,
               path=os.path.join(OUTPUT_DIR, 'model.pth')):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses':     train_losses,
        'val_losses':       val_losses,
    }, path)
    print(f"Model saved: {path}")


def load_modelCNN(path=os.path.join(OUTPUT_DIR, 'model.pth'), num_kernels=8, kernel_size=12):
    checkpoint = torch.load(path, map_location=device)
    model      = CNNMultipleLayers(num_kernels=num_kernels, kernel_size=kernel_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    train_losses = checkpoint['train_losses']
    val_losses   = checkpoint['val_losses']
    print(f"Model loaded: {path}")
    return model, train_losses, val_losses



if __name__ == '__main__':
    train_val_coverage, test_coverage = load_coverage()

    train_loader, val_loader, test_loader = get_dataloaders(
        train_val_coverage, test_coverage,
        window_size=1000,
        batch_size=1024
    )

    model     = CNNMultipleLayers(num_kernels=8, kernel_size=12, window_size=1000).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses   = []

    n_epochs = 20
    for epoch in range(1, n_epochs + 1):
        start      = time.time()
        train_loss = train_epoch(train_loader, model, poisson_loss, optimizer, epoch)
        val_loss   = validation(val_loader,   model, poisson_loss, epoch)
        elapsed    = time.time() - start

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch} took {elapsed:.1f}s — est. {elapsed * (n_epochs - epoch) / 60:.1f} min remaining")

    np.save(os.path.join(OUTPUT_DIR, 'losses.npy'), {
        'train_losses': train_losses,
        'val_losses':   val_losses
    })

    save_modelCNN(model)
    evaluate(model, test_loader, name='Test (Chr5)')