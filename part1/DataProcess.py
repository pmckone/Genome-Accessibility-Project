import os
import numpy as np
import pyBigWig
from multiprocessing import Pool, cpu_count

# --- Paths ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data', 'new_comb_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')

# --- Config ---S
BIN_SIZE         = 200
TRAIN_VAL_CHROMS = ['Chr1', 'Chr2', 'Chr3', 'Chr4']
TEST_CHROMS      = ['Chr5']


def get_srx_ids(data_dir=DATA_DIR):
    """Automatically find all SRX IDs from .bw files in the data folder."""
    srx_ids = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.bw'):
            srx_id = filename.replace('_Rep0.rpgc.bw', '').replace('.bw', '')
            srx_ids.append(srx_id)
    return sorted(srx_ids)


def find_bw_path(srx_id, data_dir=DATA_DIR):
    """Find the .bw file for a given SRX ID, trying both naming conventions."""
    candidates = [
        os.path.join(data_dir, f"{srx_id}_Rep0.rpgc.bw"),
        os.path.join(data_dir, f"{srx_id}.bw"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def extract_coverage(bw_path, chroms, bin_size=BIN_SIZE):
    """
    Extract mean read coverage in fixed-size bins for the given chromosomes.
    Returns: {chrom: np.array of mean coverage per bin}
    """
    bw = pyBigWig.open(bw_path)
    chrom_sizes = dict(bw.chroms())
    result = {}

    for chrom in chroms:
        if chrom not in chrom_sizes:
            print(f"  {chrom} not found in {os.path.basename(bw_path)}, skipping.")
            continue

        chrom_len = chrom_sizes[chrom]
        n_bins    = chrom_len // bin_size
        end       = n_bins * bin_size

        coverage  = bw.stats(chrom, 0, end, type='mean', nBins=n_bins)
        coverage  = np.array([v if v is not None else 0.0 for v in coverage],
                             dtype=np.float32)

        result[chrom] = coverage

    bw.close()
    return result


def process_sample_worker(args):
    """
    Worker function for multiprocessing — takes a single args tuple.
    Returns (srx_id, train_val_dict, test_dict) or (srx_id, None, None).
    """
    srx_id, bin_size, train_val_chroms, test_chroms = args

    bw_path = find_bw_path(srx_id)
    if bw_path is None:
        print(f"Missing: {srx_id}, skipping.")
        return srx_id, None, None

    try:
        print(f"Processing {srx_id} ({os.path.basename(bw_path)})...")
        train_val = extract_coverage(bw_path, train_val_chroms, bin_size)
        test      = extract_coverage(bw_path, test_chroms,      bin_size)
        print(f"Done: {srx_id}")
        return srx_id, train_val, test
    except Exception as e:
        print(f"Error processing {srx_id}: {e}")
        return srx_id, None, None


def process_sample(srx_id, bin_size=BIN_SIZE,
                   train_val_chroms=TRAIN_VAL_CHROMS, test_chroms=TEST_CHROMS):
    """Process a single SRX sample — returns (train_val_dict, test_dict)."""
    _, train_val, test = process_sample_worker(
        (srx_id, bin_size, train_val_chroms, test_chroms)
    )
    return train_val, test


def process_all_samples(srx_ids=None, bin_size=BIN_SIZE,
                        train_val_chroms=TRAIN_VAL_CHROMS, test_chroms=TEST_CHROMS,
                        n_workers=None):
    if srx_ids is None:
        srx_ids = get_srx_ids()
        print(f"Found {len(srx_ids)} samples\n")

    if n_workers is None:
        n_workers = cpu_count()
    print(f"Using {n_workers} workers\n")
    args = [(srx, bin_size, train_val_chroms, test_chroms) for srx in srx_ids]

    train_val_coverage = {}
    test_coverage      = {}

    with Pool(processes=n_workers) as pool:
        for srx_id, train_val, test in pool.imap_unordered(process_sample_worker, args):
            if train_val is not None:
                train_val_coverage[srx_id] = train_val
                test_coverage[srx_id]      = test

    return train_val_coverage, test_coverage


def save_coverage(train_val_coverage, test_coverage,
                  train_path=os.path.join(OUTPUT_DIR, 'train_val_coverage.npy'),
                  test_path=os.path.join(OUTPUT_DIR, 'test_coverage.npy')):

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(train_path, train_val_coverage)
    np.save(test_path,  test_coverage)
    print(f"\nSaved: {train_path}")
    print(f"Saved: {test_path}")


def load_coverage(train_path=os.path.join(OUTPUT_DIR, 'train_val_coverage.npy'),
                  test_path=os.path.join(OUTPUT_DIR, 'test_coverage.npy')):

    train_val_coverage = np.load(train_path, allow_pickle=True).item()
    test_coverage      = np.load(test_path,  allow_pickle=True).item()
    return train_val_coverage, test_coverage

def _count_windows(coverage_dict, window_size):
    total = 0
    for srx_id, chrom_dict in coverage_dict.items():
        for chrom, coverage in chrom_dict.items():
            n = len(coverage)
            if n >= window_size:
                total += n - window_size + 1
    return total
 
 
def build_windows_memmap(coverage_dict, window_size=1000,
                         windows_path=None, targets_path=None):
    print("Counting total windows (first pass)...")
    total = _count_windows(coverage_dict, window_size)
    print(f"Total windows: {total:,}  |  window_size: {window_size}")
 
    if total == 0:
        raise ValueError("No windows could be built — check coverage_dict and window_size.")

    windows_mm = np.lib.format.open_memmap(
        windows_path, mode='w+', dtype=np.float32, shape=(total, window_size)
    )
    targets_mm = np.lib.format.open_memmap(
        targets_path, mode='w+', dtype=np.float32, shape=(total,)
    )
 
    half   = window_size // 2
    offset = 0

    for srx_id, chrom_dict in coverage_dict.items():
        for chrom, coverage in chrom_dict.items():
            n = len(coverage)
            if n < window_size:
                continue
 
            n_windows = n - window_size + 1
 

            shape   = (n_windows, window_size)
            strides = (coverage.strides[0], coverage.strides[0])
            windows = np.lib.stride_tricks.as_strided(coverage, shape=shape, strides=strides)
            targets = coverage[half: half + n_windows]
 
            windows_mm[offset: offset + n_windows] = windows
            targets_mm[offset: offset + n_windows] = targets
            print(f"  {srx_id} / {chrom}: wrote {n_windows:,} windows  (total so far: {offset:,})", flush=True)
    del windows_mm
    del targets_mm
 
    print(f"\nSaved windows  → {windows_path}")
    print(f"Saved targets  → {targets_path}")
    return windows_path, targets_path
 
 
def build_windows(coverage_dict, window_size=1000):
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
 
    windows = np.concatenate(all_windows, axis=0).astype(np.float32)
    targets = np.concatenate(all_targets, axis=0).astype(np.float32)
    print(f"Built {len(windows)} windows of size {window_size}")
    return windows, targets
 
 
def save_windows(train_windows, train_targets, test_windows, test_targets,
                 output_dir=OUTPUT_DIR):
    """Save windows to disk (in-memory version)."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'train_windows.npy'), train_windows)
    np.save(os.path.join(output_dir, 'train_targets.npy'), train_targets)
    np.save(os.path.join(output_dir, 'test_windows.npy'),  test_windows)
    np.save(os.path.join(output_dir, 'test_targets.npy'),  test_targets)
    print(f"Saved windows to {output_dir}")
 
 
def load_windows(output_dir=OUTPUT_DIR, mmap_mode='r'):
    train_windows = np.load(os.path.join(output_dir, 'train_windows.npy'), mmap_mode=mmap_mode)
    train_targets = np.load(os.path.join(output_dir, 'train_targets.npy'), mmap_mode=mmap_mode)
    test_windows  = np.load(os.path.join(output_dir, 'test_windows.npy'),  mmap_mode=mmap_mode)
    test_targets  = np.load(os.path.join(output_dir, 'test_targets.npy'),  mmap_mode=mmap_mode)
    return train_windows, train_targets, test_windows, test_targets 

def build_windows_position_split(coverage_dict, window_size=200, val_fraction=0.2):
    """
    Split by genomic position within each chromosome to avoid leakage.
    First (1-val_fraction) of each chrom → train, last val_fraction → val.
    """
    train_windows, train_targets = [], []
    val_windows,   val_targets   = [], []
    half = window_size // 2

    for srx_id, chrom_dict in coverage_dict.items():
        for chrom, coverage in chrom_dict.items():
            n = len(coverage)
            if n < window_size:
                continue

            split = int(n * (1 - val_fraction))

            # Train: first 80% of chromosome
            train_cov = coverage[:split]
            n_train   = len(train_cov) - window_size + 1
            if n_train > 0:
                shape   = (n_train, window_size)
                strides = (train_cov.strides[0], train_cov.strides[0])
                w = np.lib.stride_tricks.as_strided(train_cov, shape=shape, strides=strides)
                t = train_cov[half: half + n_train]
                train_windows.append(w.copy())
                train_targets.append(t.copy())

            # Val: last 20% of chromosome
            val_cov = coverage[split:]
            n_val   = len(val_cov) - window_size + 1
            if n_val > 0:
                shape   = (n_val, window_size)
                strides = (val_cov.strides[0], val_cov.strides[0])
                w = np.lib.stride_tricks.as_strided(val_cov, shape=shape, strides=strides)
                t = val_cov[half: half + n_val]
                val_windows.append(w.copy())
                val_targets.append(t.copy())

    train_windows = np.concatenate(train_windows).astype(np.float32)
    train_targets = np.concatenate(train_targets).astype(np.float32)
    val_windows   = np.concatenate(val_windows).astype(np.float32)
    val_targets   = np.concatenate(val_targets).astype(np.float32)

    print(f"Train windows: {len(train_windows):,} | Val windows: {len(val_windows):,}")
    return train_windows, train_targets, val_windows, val_targets
if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        print(f"Data not found at {DATA_DIR}")
        print(f"Please extract new_comb_data into the data/ folder first.")
        exit(1)
 
    srx_ids = get_srx_ids()
    print(f"Found {len(srx_ids)} samples\n")
 
    train_val_coverage, test_coverage = process_all_samples(srx_ids=srx_ids, n_workers=5)
    save_coverage(train_val_coverage, test_coverage)
 
    succeeded = set(train_val_coverage.keys())
    failed    = set(srx_ids) - succeeded
 
    print(f"\n--- Summary ---")
    print(f"Total found:     {len(srx_ids)}")
    print(f"Succeeded:       {len(succeeded)}")
    print(f"Failed/Missing:  {len(failed)}")
 
    if failed:
        print(f"\nFailed samples:")
        for srx in sorted(failed):
            print(f"  - {srx}")
 
    if succeeded:
        print(f"\nSucceeded samples:")
        for srx in sorted(succeeded):
            train_chroms = list(train_val_coverage[srx].keys())
            test_chroms  = list(test_coverage[srx].keys())
            print(f"  + {srx}: train/val={train_chroms} | test={test_chroms}")
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    print("\nBuilding train/val windows (memmap)...")
    build_windows_memmap(
        train_val_coverage,
        window_size=200,
        windows_path=os.path.join(OUTPUT_DIR, 'train_windows.npy'),
        targets_path=os.path.join(OUTPUT_DIR, 'train_targets.npy'),
    )
 
    print("\nBuilding test windows (memmap)...")
    build_windows_memmap(
        test_coverage,
        window_size=200,
        windows_path=os.path.join(OUTPUT_DIR, 'test_windows.npy'),
        targets_path=os.path.join(OUTPUT_DIR, 'test_targets.npy'),
    )
 
    print("\nDone.")