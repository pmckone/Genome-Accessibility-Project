import os
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')


def load_coverage(train_path=os.path.join(OUTPUT_DIR, 'train_val_coverage.npy'),
                  test_path=os.path.join(OUTPUT_DIR, 'test_coverage.npy')):
    train_val_coverage = np.load(train_path, allow_pickle=True).item()
    test_coverage      = np.load(test_path,  allow_pickle=True).item()
    return train_val_coverage, test_coverage


def build_windows(coverage_dict, window_size=200):
    half        = window_size // 2
    all_windows = []
    all_targets = []

    for srx_id, chrom_dict in coverage_dict.items():
        for chrom, coverage in chrom_dict.items():
            if len(coverage) < window_size:
                continue

            shape   = (len(coverage) - window_size + 1, window_size)
            strides = (coverage.strides[0], coverage.strides[0])
            windows = np.lib.stride_tricks.as_strided(coverage, shape=shape, strides=strides)
            targets = coverage[half: half + len(windows)]

            all_windows.append(windows.copy())
            all_targets.append(targets.copy())

    if not all_windows:
        raise ValueError("No windows could be built. Check the coverage data and window size.")

    windows = np.concatenate(all_windows, axis=0).astype(np.float32)
    targets = np.concatenate(all_targets, axis=0).astype(np.float32)
    print(f"Built {len(windows)} windows of size {window_size}")
    return windows, targets


class CoverageDataset(Dataset):
    def __init__(self, windows, targets):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx].unsqueeze(-1)
        y = self.targets[idx].unsqueeze(0)
        return x, y


class CoverageLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(CoverageLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus(),
        )

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.regression_head(out[:, -1, :])
        return torch.clamp(out, min=1e-8, max=1e4)


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
    print(f"Epoch {epoch} | Train loss: {avg_loss:.6f} ")
    return avg_loss


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
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss    = total_loss / num_batches
    all_preds   = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    pearson_r   = np.corrcoef(all_preds, all_targets)[0, 1]

    print(f"Epoch {epoch} | Val loss: {avg_loss:.6f} | Pearson r: {pearson_r:.6f}")
    return avg_loss, pearson_r


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
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss    = total_loss / num_batches
    all_preds   = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    pearson_r   = np.corrcoef(all_preds, all_targets)[0, 1]

    print(f"\n--- {name} Evaluation ---")
    print(f"Poisson loss : {avg_loss:.6f}")
    print(f"Pearson r    : {pearson_r:.6f}")
    return avg_loss, pearson_r


def save_model_lstm(model, train_losses, val_losses, val_pearsons=None,
                    path=os.path.join(OUTPUT_DIR, 'lstm_model.pth')):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses':     train_losses,
        'val_losses':       val_losses,
        'val_pearsons':     val_pearsons if val_pearsons is not None else [],
    }, path)
    print(f"Model saved: {path}")


def load_model_lstm(path=os.path.join(OUTPUT_DIR, 'lstm_model.pth'),
                    input_dim=1, hidden_dim=64, layer_dim=2, output_dim=1):
    checkpoint = torch.load(path, map_location=device)
    model      = CoverageLSTM(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              layer_dim=layer_dim,
                              output_dim=output_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    train_losses = checkpoint['train_losses']
    val_losses   = checkpoint['val_losses']
    val_pearsons = checkpoint.get('val_pearsons', [])
    return model, train_losses, val_losses, val_pearsons


if __name__ == '__main__':
    train_val_coverage, test_coverage = load_coverage()

    all_srx = list(train_val_coverage.keys())
    rng_split = np.random.default_rng(0)
    rng_split.shuffle(all_srx)

    n_val_srx = max(1, int(len(all_srx) * 0.2))
    val_srx   = set(all_srx[:n_val_srx])
    train_srx = set(all_srx[n_val_srx:])

    train_cov = {k: v for k, v in train_val_coverage.items() if k in train_srx}
    val_cov   = {k: v for k, v in train_val_coverage.items() if k in val_srx}

    print(f"Train samples: {len(train_cov)} | Val samples: {len(val_cov)}")

    train_windows, train_targets = build_windows(train_cov, window_size=200)
    val_windows,   val_targets   = build_windows(val_cov,   window_size=200)
    test_windows,  test_targets  = build_windows(test_coverage, window_size=200)

    train_max     = train_targets.max()
    print(f"Max coverage: {train_max:.2f}")
    train_targets = train_targets / train_max
    val_targets   = val_targets   / train_max
    test_targets  = test_targets  / train_max

    rng = np.random.default_rng(42)
    idx = rng.choice(len(train_windows), size=len(train_windows) // 5, replace=False)
    train_windows = train_windows[idx].copy()
    train_targets = train_targets[idx].copy()
    print(f"Subsampled to {len(train_windows):,} training windows")

    train_dataset = CoverageDataset(train_windows, train_targets)
    val_dataset   = CoverageDataset(val_windows,   val_targets)
    test_dataset  = CoverageDataset(test_windows,  test_targets)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=512, shuffle=False,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=512, shuffle=False,
                              num_workers=8, pin_memory=True)

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    model     = CoverageLSTM(input_dim=1, hidden_dim=64, layer_dim=2, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn   = nn.PoissonNLLLoss(log_input=False, full=False)

    train_losses = []
    val_losses   = []
    val_pearson = []
    n_epochs     = 50

    best_val_loss = float('inf')
    patience      = 5
    no_improve    = 0

    for epoch in range(1, n_epochs + 1):
        start      = time.time()
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, epoch)
        val_loss, pearson_r = validation(val_loader,   model, loss_fn, epoch)
        elapsed    = time.time() - start

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_pearson.append(pearson_r)
        print(f"Epoch {epoch} took {elapsed:.1f}s — est. {elapsed * (n_epochs - epoch) / 60:.1f} min remaining")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            save_model_lstm(model, train_losses, val_losses, val_pearson)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    save_model_lstm(model, train_losses, val_losses, val_pearson)
    evaluate(model, test_loader, loss_fn=loss_fn, name='Test (Chr5)')
