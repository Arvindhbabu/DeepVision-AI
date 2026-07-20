"""
DeepVision AI

Generic Trainer
"""

import torch
from tqdm import tqdm

from src.training.base_trainer import BaseTrainer


class Trainer(BaseTrainer):

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

        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            logger=logger,
            scheduler=scheduler,
            checkpoint_manager=checkpoint_manager,
            early_stopping=early_stopping,
        )

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
        )

        for batch in progress:

            images = batch["sequence"].to(
                self.device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate(self):

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            progress = tqdm(
                self.val_loader,
                desc="Validation",
                leave=False,
            )

            for batch in progress:

                images = batch["sequence"].to(
                    self.device,
                    non_blocking=True,
                )

                labels = batch["label"].to(
                    self.device,
                    non_blocking=True,
                )

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = outputs.max(1)

                total += labels.size(0)

                correct += predicted.eq(labels).sum().item()

                progress.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def train(self, epochs):

        self.state.best_val_loss = float("inf")

        self.logger.info("Training Started")

        for epoch in range(self.state.epoch, epochs):

            self.logger.info(
                f"Epoch {epoch + 1}/{epochs}"
            )

            train_loss, train_acc = self.train_one_epoch()

            val_loss, val_acc = self.validate()

            if self.scheduler is not None:

                if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":

                    self.scheduler.step(val_loss)

                else:

                    self.scheduler.step()

            self.logger.info(
                f"Train Loss={train_loss:.4f} | "
                f"Train Acc={train_acc:.2f}% | "
                f"Val Loss={val_loss:.4f} | "
                f"Val Acc={val_acc:.2f}%"
            )

            self.history.add(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )

            if val_loss < self.state.best_val_loss:

                self.state.best_val_loss = val_loss

                self.logger.info(
                    f"New Best Model | Val Loss = {val_loss:.4f}"
                )

                self.checkpoint_manager.save_best_model(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    best_loss=self.state.best_val_loss,
                )

            self.checkpoint_manager.save_last_model(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch + 1,
                best_loss=self.state.best_val_loss,
            )

            self.state.epoch = epoch + 1

            if self.early_stopping(val_loss):

                self.logger.info(
                    "Early stopping triggered."
                )

                break

        self.checkpoint_manager.save_metrics(
            self.history.get()
        )

        return self.history.get()