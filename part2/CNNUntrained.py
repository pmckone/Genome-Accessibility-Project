import os
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Paths ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')


# --- Dataset ---
class EnhancerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx].unsqueeze(0)


def get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    train_loader = DataLoader(EnhancerDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(EnhancerDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(EnhancerDataset(X_test, y_test),
                              batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return train_loader, val_loader, test_loader



class CNN(nn.Module):
    def __init__(self, seq_len=2500, num_filters=(128, 64, 64),
                 kernel_sizes=(19, 11, 7), pool_sizes=(3, 4, 4),
                 fc_sizes=(256, 256), dropout=0.5):
        super(CNN, self).__init__()

        # Conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, num_filters[0], kernel_sizes[0], padding=kernel_sizes[0] // 2),
            nn.BatchNorm1d(num_filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(pool_sizes[0]),
        )

        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_filters[0], num_filters[1], kernel_sizes[1], padding=kernel_sizes[1] // 2),
            nn.BatchNorm1d(num_filters[1]),
            nn.ReLU(),
            nn.MaxPool1d(pool_sizes[1]),
        )

        # Conv block 3
        self.conv3 = nn.Sequential(
            nn.Conv1d(num_filters[1], num_filters[2], kernel_sizes[2], padding=kernel_sizes[2] // 2),
            nn.BatchNorm1d(num_filters[2]),
            nn.ReLU(),
            nn.MaxPool1d(pool_sizes[2]),
        )

        # Calculate flattened size after conv layers
        self.flat_size = self._get_flat_size(seq_len)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, fc_sizes[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_sizes[0], fc_sizes[1]),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_sizes[1], 1),
        )

    def _get_flat_size(self, seq_len):
        x = torch.zeros(1, 4, seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.shape[1] * x.shape[2]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# --- Loss ---
def get_loss_fn(y_train):
    """Weighted BCE to handle class imbalance."""
    pos    = y_train.sum()
    neg    = len(y_train) - pos
    weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=weight)


# --- Training ---
def train_epoch(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    total_loss  = 0
    num_batches = len(dataloader)

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss   = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch} | Train loss: {avg_loss:.6f}")
    return avg_loss


# --- Validation ---
def validation(dataloader, model, loss_fn, epoch):
    model.eval()
    total_loss  = 0
    num_batches = len(dataloader)
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y   = X.to(device), y.to(device)
            y_pred = model(X)
            total_loss += loss_fn(y_pred, y).item()
            all_preds.append(torch.sigmoid(y_pred).cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss    = total_loss / num_batches
    all_preds   = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    auprc       = average_precision_score(all_targets, all_preds)

    print(f"Epoch {epoch} | Val loss: {avg_loss:.6f} | AUPRC: {auprc:.4f}")
    return avg_loss, auprc


# --- Evaluation ---
def evaluate(model, dataloader, loss_fn, name='test'):
    model.eval()
    total_loss  = 0
    num_batches = len(dataloader)
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y   = X.to(device), y.to(device)
            y_pred = model(X)
            total_loss += loss_fn(y_pred, y).item()
            all_preds.append(torch.sigmoid(y_pred).cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss    = total_loss / num_batches
    all_preds   = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    auprc       = average_precision_score(all_targets, all_preds)

    print(f"\n--- {name} Evaluation ---")
    print(f"Loss  : {avg_loss:.6f}")
    print(f"AUPRC : {auprc:.4f}")
    return avg_loss, auprc


# --- Save / Load ---
def save_model(model, train_losses, val_losses, val_auprcs,
               path=os.path.join(OUTPUT_DIR, 'CNNUntrained.pth')):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses':     train_losses,
        'val_losses':       val_losses,
        'val_auprcs':       val_auprcs,
    }, path)
    print(f"Model saved: {path}")


def load_model(path=os.path.join(OUTPUT_DIR, 'CNNUntrained.pth')):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CNN(
    seq_len      = 2500,
    num_filters  = (128, 64, 64),
    kernel_sizes = (19, 11, 7),
    pool_sizes   = (3, 4, 4),
    fc_sizes     = (256, 256),
    dropout      = 0.5
            ).to(device)        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return (model,
            checkpoint['train_losses'],
            checkpoint['val_losses'],
            checkpoint['val_auprcs'])


# --- Main ---
if __name__ == '__main__':
    from LoadData import load_processed

    X_train, y_train, X_val, y_val, X_test, y_test = load_processed()

    train_loader, val_loader, test_loader = get_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64
    )

    model = CNN(
    seq_len      = 2500,
    num_filters  = (128, 64, 64),
    kernel_sizes = (19, 11, 7),
    pool_sizes   = (3, 4, 4),
    fc_sizes     = (256, 256),
    dropout      = 0.5
        ).to(device)
    loss_fn   = get_loss_fn(y_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    train_losses = []
    val_losses   = []
    val_auprcs   = []
    best_val_loss = float('inf')
    n_epochs     = 20

    best_auprc = 0.0

    for epoch in range(1, n_epochs + 1):
        start      = time.time()
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, epoch)
        val_loss, auprc = validation(val_loader, model, loss_fn, epoch)
        scheduler.step(val_loss)
        elapsed    = time.time() - start

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_auprcs.append(auprc)

        print(f"Epoch {epoch} took {elapsed:.1f}s — est. {elapsed * (n_epochs - epoch) / 60:.1f} min remaining")

        if auprc > best_auprc:
            best_auprc = auprc
            save_model(model, train_losses, val_losses, val_auprcs)
            print(f"  New best AUPRC: {best_auprc:.4f}")

    evaluate(model, test_loader, loss_fn, name='Test')