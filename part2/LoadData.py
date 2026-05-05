import os
import numpy as np
import pandas as pd

# --- Paths ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'datapart2')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')


SEQ_LEN = 2500  


NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def one_hot_encode(sequence, seq_len=SEQ_LEN):
    enc = np.zeros((4, seq_len), dtype=np.float32)
    for i, nuc in enumerate(sequence[:seq_len]):
        if nuc.upper() in NUC_MAP:
            enc[NUC_MAP[nuc.upper()], i] = 1.0
    return enc


def load_and_process(csv_path=os.path.join(DATA_DIR, 'arabidopsis_enhancers.csv')):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Total sequences: {len(df)}")
    print(f"Label distribution:\n{df['Label'].value_counts()}\n")

    # One-hot encode all sequences
    print("One-hot encoding sequences...")
    X = np.stack([one_hot_encode(seq) for seq in df['Sequence']])
    y = df['Label'].values.astype(np.float32)

    # Split by dataset column
    train_mask = df['dataset'] == 'Train'
    val_mask   = df['dataset'] == 'Val'
    test_mask  = df['dataset'] == 'Test'

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Train positives: {y_train.sum():.0f} ({y_train.mean()*100:.1f}%)")
    print(f"Val positives:   {y_val.sum():.0f} ({y_val.mean()*100:.1f}%)")
    print(f"Test positives:  {y_test.sum():.0f} ({y_test.mean()*100:.1f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test


def save_processed(X_train, y_train, X_val, y_val, X_test, y_test,
                   output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'),   X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'),   y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'),  X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'),  y_test)
    print(f"\nSaved to {output_dir}")


def load_processed(output_dir=OUTPUT_DIR):
    X_train = np.load(os.path.join(output_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(output_dir, 'y_train.npy'))
    X_val   = np.load(os.path.join(output_dir, 'X_val.npy'))
    y_val   = np.load(os.path.join(output_dir, 'y_val.npy'))
    X_test  = np.load(os.path.join(output_dir, 'X_test.npy'))
    y_test  = np.load(os.path.join(output_dir, 'y_test.npy'))
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    csv_path = os.path.join(DATA_DIR, 'arabidopsis_enhancers.csv')

    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        print("Please place arabidopsis_enhancers.csv in the data/ folder.")
        exit(1)

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_process(csv_path)
    save_processed(X_train, y_train, X_val, y_val, X_test, y_test)

    print("\nDone.")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}") 