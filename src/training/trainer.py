"""
DeepVision AI

Generic Trainer
"""

from pathlib import Path

import torch


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        logger,
        checkpoint_manager,
        early_stopping,
        scheduler=None,
    ):

        self.model = model.to(device)

        self.optimizer = optimizer

        self.criterion = criterion

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.device = device

        self.logger = logger

        self.checkpoint = checkpoint_manager

        self.early_stopping = early_stopping

        self.scheduler = scheduler

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for batch in self.train_loader:

            images = batch["sequence"].to(self.device)

            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader)

        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def validate(self):

        self.model.eval()

        running_loss = 0.0

        correct = 0

        total = 0

        with torch.no_grad():

            for batch in self.val_loader:

                images = batch["sequence"].to(self.device)

                labels = batch["label"].to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = outputs.max(1)

                total += labels.size(0)

                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.val_loader)

        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def train(self, epochs):

        best_loss = float("inf")

        history = []

        self.logger.info("Training Started")

        for epoch in range(epochs):

            train_loss, train_acc = self.train_one_epoch()

            val_loss, val_acc = self.validate()

            if self.scheduler:

                self.scheduler.step()

            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss={train_loss:.4f} | "
                f"Train Acc={train_acc:.2f}% | "
                f"Val Loss={val_loss:.4f} | "
                f"Val Acc={val_acc:.2f}%"
            )

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

            if val_loss < best_loss:

                best_loss = val_loss

                self.checkpoint.save_best_model(self.model)

            self.checkpoint.save_last_model(self.model)

            if self.early_stopping(val_loss):

                self.logger.info("Early stopping triggered.")

                break

        self.checkpoint.save_metrics(history)

        return history