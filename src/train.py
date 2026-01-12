import torch
import torch.nn as nn
import hydra
from src.utils.factory import get_model
from src.utils.logger import setup_wandb

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg):
    run = setup_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        model.train()
    
        inputs = torch.randn(cfg.batch_size, 10, cfg.models.input_dim).to(device)
        targets = torch.randn(cfg.batch_size, cfg.models.output_dim).to(device)

        outputs, _, _ = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run.log({"train/loss": loss.item(), "epoch": epoch})
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    
    run.finish()

if __name__ == "__main__":
    train()