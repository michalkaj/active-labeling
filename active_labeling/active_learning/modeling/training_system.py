from typing import Optional, Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.metrics import Metric, Accuracy
from torch.nn import functional as F
from torch.optim import Adam

from active_labeling.active_learning.modeling.wrappers import BAYESIAN_SAMPLE_DIM, \
    MonteCarloWrapper


class TrainingSystem(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 learning_rate: float = 1e-3,
                 test_sample_size: Optional[int] = None,
                 loss: Callable = F.cross_entropy):
        super().__init__()
        self._model = model
        self._loss = loss
        self.metrics = {
            'loss': _Loss(),
            'accuracy': Accuracy(),
        }
        self._learning_rate = learning_rate
        self._test_sample_size = test_sample_size

    def forward(self, x, **kwargs):
        return self._model(x, **kwargs)

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        logits = self.forward(images, sample_size=1).squeeze(BAYESIAN_SAMPLE_DIM)
        loss = self._loss(logits, labels)

        y_pred = logits.argmax(-1)
        self.log(
            'train_accuracy',
            self.metrics['accuracy'](y_pred.detach().cpu(), labels.detach().cpu()),
            on_step=False,
            on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        with torch.no_grad():
            if self._test_sample_size is None:
                if isinstance(self._model, MonteCarloWrapper):
                    logits = self.forward(images, deterministic=True)
                else:
                    logits = self.forward(images)
            else:
                logits = self.forward(images).mean(dim=BAYESIAN_SAMPLE_DIM)
            loss = self._loss(logits, labels, reduction='none')

        y_pred = logits.argmax(-1)

        self.log(
            'loss',
            self.metrics['loss'](loss.detach().cpu()),
            on_step=False,
            on_epoch=True
        )
        self.log(
            'val_accuracy',
            self.metrics['accuracy'](y_pred.detach().cpu(), labels.detach().cpu()),
            on_step=False,
            on_epoch=True
        )

    def configure_optimizers(self):
        return Adam(self._model.parameters(), lr=self._learning_rate)


class _Loss(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('loss', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, loss: torch.Tensor):
        self.loss += torch.sum(loss)
        self.total += len(loss)

    def compute(self):
        return self.loss.float() / self.total

class AccuracyEarlyStopping(EarlyStopping):
    def __init__(self, monitor, threshold=0.98):
        super().__init__(
            monitor,
        )
        self._threshold = threshold

    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_end(self, trainer, pl_module):
        logs = trainer.logger_connector.callback_metrics
        current = logs.get(self.monitor)
        if current.compute() >= self._threshold:
            trainer.should_stop = True