import argparse
import json
from pathlib import Path

import torch

from sofnet.config import load_config
from sofnet.data import build_dataloader
from sofnet.engine import load_checkpoint, save_checkpoint, train_one_epoch, validate_loss
from sofnet.losses import SOFLoss
from sofnet.model import SOFNet
from sofnet.utils import cosine_warmup_lambda, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    train_loader = build_dataloader(config, "train", shuffle=True, augment=True)
    validation_loader = build_dataloader(config, "validation", shuffle=False, augment=False)
    model = SOFNet(config).to(device)
    criterion = SOFLoss(config).to(device)
    training = config["training"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=training["learning_rate"],
        momentum=training["momentum"],
        weight_decay=training["weight_decay"],
    )
    total_steps = training["epochs"] * len(train_loader)
    warmup_steps = training["warmup_epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_warmup_lambda(step, total_steps, warmup_steps)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_loss = float("inf")
    resume = training.get("resume", "Real Filepath")
    if resume != "Real Filepath" and Path(resume).is_file():
        checkpoint = load_checkpoint(resume, model, optimizer, scheduler, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))
    history = []
    for epoch in range(start_epoch, training["epochs"]):
        train_metrics = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            training["gradient_clip"],
        )
        validation_metrics = validate_loss(model, criterion, validation_loader, device)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        print(json.dumps(record))
        save_checkpoint(
            output_dir / "last.pt", model, optimizer, scheduler, epoch, config, best_loss
        )
        if validation_metrics["total"] < best_loss:
            best_loss = validation_metrics["total"]
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, scheduler, epoch, config, best_loss
            )
        with (output_dir / "history.json").open("w", encoding="utf-8") as stream:
            json.dump(history, stream, indent=2)


if __name__ == "__main__":
    main()
