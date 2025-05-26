import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from argparse import Namespace
from torch.amp import autocast, GradScaler
from loguru import logger

script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'model'))


parser = argparse.ArgumentParser(description="Train RibonanzaNet model")
parser.add_argument('--pairwise_config', type=str, default='configs/pairwise.yaml')
parser.add_argument('--train_config',    type=str, default='configs/training.yaml')
parser.add_argument('--train_data',      type=str, default='data/train_sequences.csv')
parser.add_argument('--train_labels',    type=str, default='data/train_labels.csv')
parser.add_argument('--valid_data',      type=str, default='data/valid_sequences.csv')
parser.add_argument('--valid_labels',    type=str, default='data/valid_labels.csv')
parser.add_argument('--checkpoint',      type=str, default=None)
parser.add_argument('--log_dir',         type=str, default='logs')
parser.add_argument('--save_dir',        type=str, default='model/checkpoints')
args = parser.parse_args()


logger.remove()

logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}"
)

log_file = os.path.join(log_dir, "train.log")
logger.add(
    log_file,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    rotation="10 MB",
    enqueue=True,
    encoding="utf-8"

def abs_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(project_root, path)

pair_cfg_path   = abs_path(args.pairwise_config)
train_cfg_path  = abs_path(args.train_config)
train_data_path = abs_path(args.train_data)
train_lbl_path  = abs_path(args.train_labels)
valid_data_path = abs_path(args.valid_data)
valid_lbl_path  = abs_path(args.valid_labels)
log_dir         = abs_path(args.log_dir)
save_dir        = abs_path(args.save_dir)
checkpoint_path = abs_path(args.checkpoint) if args.checkpoint else None

os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# TensorBoard
writer = SummaryWriter(log_dir=log_dir)


from model.Network import RibonanzaNet


with open(pair_cfg_path)  as f: pairwise_cfg = yaml.safe_load(f)
with open(train_cfg_path) as f: train_cfg    = yaml.safe_load(f)
cfg_pw = Namespace(**pairwise_cfg)
cfg_tr = Namespace(**train_cfg)

# enforce output dim = 3 for (x,y,z)
cfg_pw.nclass = 3


vocab = {"A":0, "C":1, "G":2, "U":3}
pad_id = len(vocab)          # 4
cfg_pw.ntoken = pad_id + 1   # 5
max_len = cfg_pw.max_len

def tokenize(seq: str) -> torch.LongTensor:
    ids = [vocab[ch] for ch in seq]
    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)

class RNADataset(Dataset):
    def __init__(self, seqs_df: pd.DataFrame, coords_dict: dict):
        self.seqs   = seqs_df.reset_index(drop=True)
        self.coords = coords_dict

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        row     = self.seqs.iloc[idx]
        seq_ids = row['seq_ids']
        xyz     = self.coords[row['target_id']]
        return seq_ids, xyz

def load_dataset(seq_path: str, label_path: str,
                 batch_size: int, shuffle: bool=False):
    seqs_df   = pd.read_csv(seq_path)
    labels_df = pd.read_csv(label_path)

    seqs_df['seq_ids']     = seqs_df['sequence'].map(tokenize)
    labels_df['target_id'] = labels_df['ID'].str.rsplit('_', n=1).str[0]

    coords = {}
    for tid, grp in labels_df.groupby('target_id'):
        grp = grp.sort_values('resid')
        xyz = grp[['x_1','y_1','z_1']].to_numpy(np.float32)
        if xyz.shape[0] < max_len:
            pad = np.zeros((max_len - xyz.shape[0], 3), np.float32)
            xyz = np.vstack([xyz, pad])
        else:
            xyz = xyz[:max_len]
        coords[tid] = torch.tensor(xyz, dtype=torch.float32)

    ds = RNADataset(seqs_df, coords)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=shuffle
    )
    return loader

train_loader = load_dataset(train_data_path, train_lbl_path,
                            cfg_tr.batch_size, shuffle=True)
val_loader   = load_dataset(valid_data_path, valid_lbl_path,
                            cfg_tr.test_batch_size, shuffle=False)

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = RibonanzaNet(cfg_pw).to(device)
optimizer = optim.AdamW(
    model.parameters(),
    lr=cfg_tr.learning_rate,
    weight_decay=cfg_tr.weight_decay
)
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=cfg_tr.lr_step,
    gamma=cfg_tr.lr_gamma
)
criterion = nn.MSELoss()
scaler    = GradScaler()

# resume if checkpoint
start_epoch = 1
if checkpoint_path:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    logger.info(f"Resumed from checkpoint {checkpoint_path} at epoch {ckpt['epoch']}")

best_val = float('inf')

for epoch in range(start_epoch, cfg_tr.epochs + 1):
    model.train()
    total_train = 0.0

    for seq_ids, coords_true in train_loader:
        seq_ids, coords_true = seq_ids.to(device), coords_true.to(device)
        src_mask = torch.ones_like(seq_ids, dtype=torch.long, device=device)

        optimizer.zero_grad()
        with autocast():
            pred = model(seq_ids, src_mask=src_mask)
            loss = criterion(pred, coords_true)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg_tr.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_train += loss.item()

    avg_train = total_train / len(train_loader)
    writer.add_scalar('Loss/Train', avg_train, epoch)

    # validation
    model.eval()
    total_val = 0.0
    with torch.no_grad():
        for seq_ids, coords_true in val_loader:
            seq_ids, coords_true = seq_ids.to(device), coords_true.to(device)
            src_mask = torch.ones_like(seq_ids, dtype=torch.long, device=device)
            pred = model(seq_ids, src_mask=src_mask)
            total_val += criterion(pred, coords_true).item()

    avg_val = total_val / len(val_loader)
    writer.add_scalar('Loss/Val', avg_val, epoch)

    logger.info(f"Epoch {epoch}/{cfg_tr.epochs} — Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    if avg_val < best_val:
        best_val = avg_val
        ckpt_file = os.path.join(save_dir, f'best_epoch{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val
        }, ckpt_file)
        logger.info(f"Saved new best model to {ckpt_file}")

    scheduler.step()

writer.close()
logger.info("Training complete.")
