import torch
import torch.nn as nn
import hydra
from tqdm import tqdm

from utils.factory import get_model
from utils.logger import setup_wandb
from data import create_dataloaders


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg):
    run = setup_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading data for {cfg.dataset.ticker}...")
    train_loader, val_loader, test_loader, pipeline = create_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}\n")

    model = get_model(cfg).to(device)
    loss_fn = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_loss = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, _, _ = model(inputs)
            outputs = outputs.squeeze(-1)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs, _, _ = model(inputs)
                outputs = outputs.squeeze(-1)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        run.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "epoch": epoch
        })

        print(f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  -> New best validation loss!")

    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, _, _ = model(inputs)
            outputs = outputs.squeeze(-1)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"\nFinal Test Loss: {test_loss:.6f}")
    run.log({"test/loss": test_loss})

    run.finish()


if __name__ == "__main__":
    train()
