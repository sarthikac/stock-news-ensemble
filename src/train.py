import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from datasets.merged_dataset import MergedDataset
from models.ensemble import EnsembleModel
from data.news_processing import encode_headlines
from sklearn.metrics import accuracy_score

load_dotenv()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    # collate tech tensors, texts, labels
    tech = torch.stack([b['tech'] for b in batch], dim=0)
    texts = [b['text'] for b in batch]
    labels = torch.stack([b['label'] for b in batch], dim=0)
    dates = [b['date'] for b in batch]
    return tech, texts, labels, dates

def train_epoch(model, loader, opt, device, loss_fn):
    model.train()
    losses = []
    all_preds = []
    all_labels = []
    for tech, texts, labels, _ in tqdm(loader, desc="train"):
        enc = encode_headlines(texts, device=device)
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']
        tech = tech.to(device)
        labels = labels.to(device)
        opt.zero_grad()
        logits = model(tech, input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    return np.mean(losses), accuracy_score(all_labels, all_preds)

def eval_epoch(model, loader, device, loss_fn):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for tech, texts, labels, _ in tqdm(loader, desc="eval"):
            enc = encode_headlines(texts, device=device)
            input_ids = enc['input_ids']
            attention_mask = enc['attention_mask']
            tech = tech.to(device)
            labels = labels.to(device)
            logits = model(tech, input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            losses.append(loss.item())
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return np.mean(losses), accuracy_score(all_labels, all_preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price_csv", required=True)
    parser.add_argument("--news_csv", default=None)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device

    # dataset
    ds = MergedDataset(args.price_csv, news_csv=args.news_csv, seq_len=args.seq_len, device=device)
    n = len(ds)
    train_n = int(n * 0.8)
    val_n = int(n * 0.1)
    test_n = n - train_n - val_n
    train_set, val_set, test_set = torch.utils.data.random_split(ds, [train_n, val_n, test_n], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # model
    n_features = len(ds.feature_cols)
    model = EnsembleModel(n_features=n_features, lstm_hidden=64, text_model_name="distilbert-base-uncased", combine_hidden=128, freeze_text=False)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, opt, device, loss_fn)
        val_loss, val_acc = eval_epoch(model, val_loader, device, loss_fn)
        print(f"train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved best_model.pt")

    # final test
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_acc = eval_epoch(model, test_loader, device, loss_fn)
    print(f"FINAL TEST loss {test_loss:.4f} acc {test_acc:.4f}")

    # Save feature column list for inference/backtest
    import json
    with open("feature_cols.json", "w") as f:
        json.dump(ds.feature_cols, f)
    print("Saved feature_cols.json")

if __name__ == "__main__":
    main()
