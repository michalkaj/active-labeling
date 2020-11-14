from typing import Dict, Optional, Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Metric
from torch.nn import functional as F
from torch.optim import Adam

from active_labeling.active_learning.models.monte_carlo_wrapper import BAYESIAN_SAMPLE_DIM, \
    MonteCarloWrapper


class TrainingSystem(pl.LightningModule):
    def __init__(self,
                 model: MonteCarloWrapper,
                 metrics: Dict[str, Metric],
                 learning_rate: float = 1e-3,
                 test_sample_size: Optional[int] = None,
                 loss: Callable = F.cross_entropy):
        super().__init__()
        self._model = model
        self._loss = loss
        self.metrics = metrics
        self.metrics['loss'] = _Loss()
        self._learning_rate = learning_rate
        self._test_sample_size = test_sample_size

    def forward(self, x, **kwargs):
        return self._model(x, **kwargs)

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        logits = self.forward(images, sample_size=1).squeeze(BAYESIAN_SAMPLE_DIM)
        loss = self._loss(logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        with torch.no_grad():
            if self._test_sample_size is None:
                kwargs = {'sample_size': 1, 'deterministic': True}
            else:
                kwargs = {'sample_size': self._test_sample_size}
            logits = self.forward(
                images, **kwargs).mean(dim=BAYESIAN_SAMPLE_DIM)
            loss = self._loss(logits, labels, reduction='none')

        y_pred = logits.argmax(-1)

        self.log('loss', self.metrics['loss'](loss.detach().cpu().numpy()), on_step=False, on_epoch=True)
        self.log('accuracy', self.metrics['accuracy'](y_pred.detach().cpu().numpy(), labels.detach().cpu().numpy()), on_step=False, on_epoch=True)

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
