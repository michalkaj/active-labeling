from typing import Dict

import pytorch_lightning as pl
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
        x, y = batch
        logits = self.forward(x)
        loss = self._loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self._loss(logits, y)

        y_pred = logits.argmax(-1)
        for metric in self.metrics.values():
            metric.update(y_pred.detach().cpu(), y.detach().cpu())

        return {'val_loss_batch': loss}

    def on_validation_epoch_start(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def validation_epoch_end(self, batches):
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def configure_optimizers(self):
        return Adam(self._model.parameters(), lr=self._learning_rate)
