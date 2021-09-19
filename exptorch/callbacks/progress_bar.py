from tqdm import tqdm

from exptorch.callbacks.callbacks import Callback


class ProgressBar(Callback):
    def __init__(self, refresh_rate: int = 25):
        self.main_progress_bar = None
        self.val_progress_bar = None
        self.refresh_rate = refresh_rate

    @staticmethod
    def init_train_progress_bar() -> tqdm:
        return tqdm(desc="Training")

    @staticmethod
    def init_val_progress_bar() -> tqdm:
        return tqdm(desc="Validation")

    def on_train_start(self, trainer):
        self.main_progress_bar = self.init_train_progress_bar()

    def on_epoch_start(self, trainer):
        total_num_batches = trainer.state.num_train_batches + trainer.state.num_val_batches
        self.main_progress_bar.reset(total_num_batches)
        self.main_progress_bar.set_description(
            f"Epoch {trainer.state.epoch} / {trainer.max_epochs}"
        )

    def on_batch_end(self, trainer, batch, batch_idx, loss, model_output):
        if trainer.state.batch % self.refresh_rate == 0:
            self.main_progress_bar.update(self.refresh_rate)
            # @ TODO format loss output
            self.main_progress_bar.set_postfix(loss=loss)

    def on_train_end(self, trainer):
        self.main_progress_bar.close()
