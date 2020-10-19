from typing import Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Metric, Accuracy
from torch import nn
from torch.optim import Adam


class TrainingSystem(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 metrics: Dict[str, Metric],
                 learning_rate: float = 1e-3):
        super().__init__()
        self._model = model
        self._loss = nn.CrossEntropyLoss()
        self.metrics = metrics
        self._learning_rate = learning_rate

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        logits = self.forward(images)
        loss = self._loss(logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        with torch.no_grad():
            logits = self.forward(images)
            loss = self._loss(logits, labels)

        y_pred = logits.argmax(-1)
        for metric in self.metrics.values():
            metric.update(y_pred.detach().cpu(), labels.detach().cpu())

        return {'val_loss_batch': loss}

    def on_train_epoch_start(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def validation_epoch_end(self, batches):
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def configure_optimizers(self):
        return Adam(self._model.parameters(), lr=self._learning_rate)
