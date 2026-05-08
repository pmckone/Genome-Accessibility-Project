import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, accuracy_score
import os
import copy
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Paths ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')

class TransferCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=12, padding=12 // 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Conv1d(32, 32, kernel_size=12, padding=12 // 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Conv1d(32, 32, kernel_size=12, padding=12 // 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        center = x.shape[2] // 2
        x = x[:, :, center]
        return self.classifier(x)
    
    def train_model(self, train_loader, val_loader, loss_fn, device,
                epochs=20, lr=1e-5, weight_decay=1e-4, patience=5,
                save_path=os.path.join(OUTPUT_DIR, 'TransferCNN.pth')):

        self.to(device)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        train_losses = []
        val_losses = []
        val_auprcs = []

        best_val_auprc = 0.0
        no_improve = 0

        for epoch in range(1, epochs + 1):
            start = time.time()

            self.train()
            total_train_loss = 0

            for X, y in train_loader:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                logits = self(X)
                loss = loss_fn(logits, y)

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.eval()
            total_val_loss = 0
            all_probs = []
            all_targets = []

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)

                    logits = self(X)
                    loss = loss_fn(logits, y)
                    total_val_loss += loss.item()

                    probs = torch.sigmoid(logits)
                    all_probs.append(probs.cpu().numpy())
                    all_targets.append(y.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            all_probs = np.concatenate(all_probs).flatten()
            all_targets = np.concatenate(all_targets).flatten()

            val_auprc = average_precision_score(all_targets, all_probs)

            val_losses.append(avg_val_loss)
            val_auprcs.append(val_auprc)

            elapsed = time.time() - start

            print(
                f"Epoch {epoch} | "
                f"Train loss: {avg_train_loss:.6f} | "
                f"Val loss: {avg_val_loss:.6f} | "
                f"Val AUPRC: {val_auprc:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                no_improve = 0
                save_transfer_model(
                    self,
                    train_losses,
                    val_losses,
                    val_auprcs,
                    path=save_path
                )

                print(f"  New best validation AUPRC: {best_val_auprc:.4f}")

            else:
                no_improve += 1

                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return train_losses, val_losses, val_auprcs

def evaluate_transfer(model, dataloader, loss_fn, device,
                      name="Transfer Test"):

    model.eval()

    total_loss = 0
    num_batches = len(dataloader)

    all_probs = []
    all_targets = []

    with torch.no_grad():

        for X, y in dataloader:

            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            loss = loss_fn(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / num_batches

    all_probs = np.concatenate(all_probs).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    auprc = average_precision_score(all_targets, all_probs)

    preds_binary = (all_probs >= 0.5).astype(int)
    accuracy = accuracy_score(all_targets, preds_binary)

    print(f"\n--- {name} Evaluation ---")
    print(f"BCE loss : {avg_loss:.6f}")
    print(f"AUPRC    : {auprc:.4f}")
    print(f"Accuracy : {accuracy:.4f}")

    return (
        avg_loss,
        auprc,
        accuracy,
        all_probs,
        all_targets
    )

def save_transfer_model(model, train_losses, val_losses, val_auprcs,
                        path=os.path.join(OUTPUT_DIR, 'TransferCNN.pth')):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_auprcs': val_auprcs,
    }, path)

    print(f"Transfer model saved: {path}")

def load_transfer_model(path=os.path.join(OUTPUT_DIR, 'TransferCNN.pth')):

    checkpoint = torch.load(path, map_location=device)

    model = TransferCNN().to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    return (
        model,
        checkpoint['train_losses'],
        checkpoint['val_losses'],
        checkpoint['val_auprcs']
    )

def get_transfer_loss_fn(y_train, device):

    pos = y_train.sum()
    neg = len(y_train) - pos

    pos_weight = torch.tensor(
        [neg / pos],
        dtype=torch.float32
    ).to(device)

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)