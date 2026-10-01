"""Early stopping on (train-loss plateau AND test-score decline), with epoch guards."""
from __future__ import annotations


class EarlyStopper:
    """Call update(train_loss, score) once per finished epoch.

    Stop when BOTH hold (and epoch >= min_epochs):
      (a) relative train-loss improvement over the last `n` epochs < `min_rel_impr`:
          (loss[t-n] - loss[t]) / loss[t-n] < min_rel_impr
      (b) test score is below its best AND below its value `n` epochs ago.
    Also stop at max_epochs. `reason` is set when stopped.
    """

    def __init__(self, n: int = 3, min_rel_impr: float = 0.02, min_epochs: int = 8, max_epochs: int = 60):
        self.n, self.min_rel_impr = n, min_rel_impr
        self.min_epochs, self.max_epochs = min_epochs, max_epochs
        self.losses: list[float] = []
        self.scores: list[float] = []
        self.best_score = float("-inf")
        self.best_epoch = 0
        self.reason: str | None = None

    @property
    def epoch(self) -> int:
        return len(self.losses)

    def update(self, train_loss: float, score: float) -> bool:
        self.losses.append(float(train_loss))
        self.scores.append(float(score))
        e = self.epoch
        if score > self.best_score:
            self.best_score, self.best_epoch = float(score), e
        if e >= self.max_epochs:
            self.reason = f"max_epochs ({self.max_epochs})"
            return True
        if e < self.min_epochs or e <= self.n:
            return False
        old_loss = self.losses[-1 - self.n]
        rel = (old_loss - self.losses[-1]) / max(abs(old_loss), 1e-12)
        plateau = rel < self.min_rel_impr
        worse = score < self.best_score and score < self.scores[-1 - self.n]
        if plateau and worse:
            self.reason = (f"early_stop: train loss gain {rel:.3%} < {self.min_rel_impr:.0%} over {self.n} epochs "
                           f"and test score {score:.4f} < best {self.best_score:.4f} (epoch {self.best_epoch}) "
                           f"and < {self.scores[-1 - self.n]:.4f} ({self.n} epochs ago)")
            return True
        return False

    def state_dict(self):
        return dict(losses=self.losses, scores=self.scores, best_score=self.best_score,
                    best_epoch=self.best_epoch, reason=self.reason)

    def load_state_dict(self, d):
        self.losses, self.scores = list(d["losses"]), list(d["scores"])
        self.best_score, self.best_epoch, self.reason = d["best_score"], d["best_epoch"], d["reason"]
