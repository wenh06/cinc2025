"""
"""

import argparse
import logging
import os
import sys
import textwrap  # noqa: F401
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np  # noqa: F401
import torch
from torch import nn
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch.utils.data import DataLoader, Dataset
from torch_ecg.cfg import CFG
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.utils.misc import get_date_str, str2bool
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from tqdm.auto import tqdm

from cfg import ModelCfg, TrainCfg
from const import MODEL_CACHE_DIR
from dataset import CINC2025Dataset
from models import CRNN_CINC2025
from models.crnn import make_safe_globals
from outputs import CINC2025Outputs
from utils.scoring_metrics import compute_challenge_metrics  # noqa: F401

os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)

__all__ = [
    "CINC2025Trainer",
]


class CINC2025Trainer(BaseTrainer):
    """Trainer for the CinC2025 challenge.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained
    model_config : dict
        The configuration of the model,
        used to keep a record in the checkpoints
    train_config : dict
        The configuration of the training,
        including configurations for the data loader, for the optimization, etc.
        will also be recorded in the checkpoints.
        `train_config` should at least contain the following keys:

            - "monitor": obj:`str`,
            - "loss": obj:`str`,
            - "n_epochs": obj:`int`,
            - "batch_size": obj:`int`,
            - "learning_rate": obj:`float`,
            - "lr_scheduler": obj:`str`,
            - "lr_step_size": obj:`int`, optional, depending on the scheduler
            - "lr_gamma": obj:`float`, optional, depending on the scheduler
            - "max_lr": obj:`float`, optional, depending on the scheduler
            - "optimizer": obj:`str`,
            - "decay": obj:`float`, optional, depending on the optimizer
            - "momentum": obj:`float`, optional, depending on the optimizer

    device : torch.device, optional
        The device to be used for training
    lazy : bool, default True
        Whether to initialize the data loader lazily

    """

    __DEBUG__ = True
    __name__ = "CINC2025Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
        train_config["classes"] = train_config["chagas_classes"]
        super().__init__(
            model=model,
            dataset_cls=CINC2025Dataset,
            model_config=model_config,
            train_config=train_config,
            device=device,
            lazy=lazy,
        )
        if hasattr(self._model.config, "monitor") and self._model.config.monitor is not None:
            self._train_config["monitor"] = self._model.config.monitor

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
        """
        setup the dataloaders for training and validation

        Parameters
        ----------
        train_dataset: Dataset, optional,
            the training dataset
        val_dataset: Dataset, optional,
            the validation dataset

        """
        if train_dataset is None:
            train_dataset = self.dataset_cls(
                config=self.train_config,
                training=True,
                lazy=True,
            )

        if self.train_config.debug:
            val_train_dataset = train_dataset
        else:
            val_train_dataset = None
        # if val_dataset is None:
        #     val_dataset = self.dataset_cls(
        #         config=self.train_config,
        #         training=False,
        #         lazy=True,
        #     )

        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
        if self.device == torch.device("cpu"):
            num_workers = 1
        else:
            num_workers = 4

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        if self.train_config.debug:
            self.val_train_loader = DataLoader(
                dataset=val_train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            self.val_train_loader = None
        if val_dataset is None:
            self.val_loader = None
        else:
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )

    def train(self) -> OrderedDict:
        """Train the model.

        Returns
        -------
        best_state_dict : OrderedDict
            The state dict of the best model.

        """
        self._setup_optimizer()

        self._setup_scheduler()

        self._setup_criterion()

        if self.train_config.monitor is not None:
            # if monitor is set but val_loader is None, use train_loader for validation
            # and choose the best model based on the metrics on the train set
            if self.val_loader is None and self.val_train_loader is None:
                self.val_train_loader = self.train_loader
                self.log_manager.log_message(
                    (
                        "No separate validation set is provided, while monitor is set. "
                        "The training set will be used for validation, "
                        "and the best model will be selected based on the metrics on the training set"
                    ),
                    level=logging.WARNING,
                )

        msg = textwrap.dedent(
            f"""
            Starting training:
            ------------------
            Epochs:          {self.n_epochs}
            Batch size:      {self.batch_size}
            Learning rate:   {self.lr}
            Training size:   {self.n_train}
            Validation size: {self.n_val}
            Device:          {self.device.type}
            Optimizer:       {self.train_config.optimizer}
            Dataset classes: {self.train_config.classes}
            -----------------------------------------
            """
        )
        self.log_manager.log_message(msg)

        start_epoch = self.epoch
        for _ in range(start_epoch, self.n_epochs):
            # train one epoch
            self.model.train()
            self.epoch_loss = 0
            with tqdm(
                total=self.n_train,
                desc=f"Epoch {self.epoch}/{self.n_epochs}",
                unit="signals",
                dynamic_ncols=True,
                mininterval=1.0,
            ) as pbar:
                self.log_manager.epoch_start(self.epoch)
                # train one epoch
                self.train_one_epoch(pbar)

                # evaluate on train set, if debug is True
                if self.val_train_loader is not None:
                    eval_train_res = self.evaluate(self.val_train_loader)
                    self.log_manager.log_metrics(
                        metrics=eval_train_res,
                        step=self.global_step,
                        epoch=self.epoch,
                        part="train",
                    )
                else:
                    eval_train_res = {}
                # evaluate on val set
                if self.val_loader is not None:
                    eval_res = self.evaluate(self.val_loader)
                    self.log_manager.log_metrics(
                        metrics=eval_res,
                        step=self.global_step,
                        epoch=self.epoch,
                        part="val",
                    )
                elif self.val_train_loader is not None:
                    # if no separate val set, use the metrics on the train set
                    eval_res = eval_train_res
                else:
                    eval_res = {}

                # update best model and best metric if monitor is set
                if self.train_config.monitor is not None:
                    if eval_res[self.train_config.monitor] > self.best_metric:
                        self.best_metric = eval_res[self.train_config.monitor]
                        self.best_state_dict = self._model.state_dict()
                        self.best_eval_res = deepcopy(eval_res)
                        self.best_epoch = self.epoch
                        self.pseudo_best_epoch = self.epoch
                    elif self.train_config.early_stopping:
                        if eval_res[self.train_config.monitor] >= self.best_metric - self.train_config.early_stopping.min_delta:
                            self.pseudo_best_epoch = self.epoch
                        elif self.epoch - self.pseudo_best_epoch >= self.train_config.early_stopping.patience:
                            msg = f"early stopping is triggered at epoch {self.epoch}"
                            self.log_manager.log_message(msg)
                            break

                    msg = textwrap.dedent(
                        f"""
                        best metric = {self.best_metric},
                        obtained at epoch {self.best_epoch}
                    """
                    )
                    self.log_manager.log_message(msg)

                    # save checkpoint
                    save_suffix = f"epochloss_{self.epoch_loss:.5f}_metric_{eval_res[self.train_config.monitor]:.2f}"
                else:
                    save_suffix = f"epochloss_{self.epoch_loss:.5f}"
                save_filename = f"{self.save_prefix}_epoch{self.epoch}_{get_date_str()}_{save_suffix}.pth.tar"
                save_path = self.train_config.checkpoints / save_filename
                if self.train_config.keep_checkpoint_max != 0:
                    self.save_checkpoint(str(save_path))
                    self.saved_models.append(save_path)
                # remove outdated models
                if len(self.saved_models) > self.train_config.keep_checkpoint_max > 0:
                    model_to_remove = self.saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except Exception:
                        self.log_manager.log_message(f"failed to remove {str(model_to_remove)}")

                # update learning rate using lr_scheduler
                if self.train_config.lr_scheduler.lower() == "plateau":
                    self._update_lr(eval_res)

                self.log_manager.epoch_end(self.epoch)

            self.epoch += 1

        # save the best model
        if self.best_metric > -np.inf:
            if self.train_config.final_model_name:
                save_filename = self.train_config.final_model_name
            else:
                save_suffix = f"metric_{self.best_eval_res[self.train_config.monitor]:.2f}"
                save_filename = f"BestModel_{self.save_prefix}{self.best_epoch}_{get_date_str()}_{save_suffix}.pth.tar"
            save_path = self.train_config.model_dir / save_filename
            # self.save_checkpoint(path=str(save_path))
            self._model.save(path=str(save_path), train_config=self.train_config)
            self.log_manager.log_message(f"best model is saved at {save_path}")
        elif self.train_config.monitor is None:
            self.log_manager.log_message("no monitor is set, the last model is selected and saved as the best model")
            self.best_state_dict = self._model.state_dict()
            save_filename = f"BestModel_{self.save_prefix}{self.epoch}_{get_date_str()}.pth.tar"
            save_path = self.train_config.model_dir / save_filename
            # self.save_checkpoint(path=str(save_path))
            self._model.save(path=str(save_path), train_config=self.train_config)
        else:
            raise ValueError("No best model found!")

        self.log_manager.close()

        if not self.best_state_dict:
            # in case no best model is found,
            # e.g. monitor is not set, or keep_checkpoint_max is 0
            self.best_state_dict = self._model.state_dict()

        return self.best_state_dict

    def train_one_epoch(self, pbar: tqdm) -> None:
        """Train one epoch, and update the progress bar

        Parameters
        ----------
        pbar : tqdm
            the progress bar for training

        """
        for epoch_step, input_tensors in enumerate(self.train_loader):
            self.global_step += 1
            n_samples = input_tensors["signals"].shape[self.batch_dim]

            out_tensors = self.run_one_step(input_tensors)

            # NOTE: loss is computed in the model, and kept in the out_tensors
            loss = out_tensors["chagas_loss"]

            if self.train_config.flooding_level > 0:
                flood = (loss - self.train_config.flooding_level).abs() + self.train_config.flooding_level
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                flood.backward()
            else:
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
            self.optimizer.step()
            self._update_lr()

            if self.global_step % self.train_config.log_step == 0:
                train_step_metrics = {"loss": loss.item()}
                if self.scheduler:
                    train_step_metrics.update({"lr": self.scheduler.get_last_lr()[0]})
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                            "lr": self.scheduler.get_last_lr()[0],
                        }
                    )
                else:
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                        }
                    )
                if self.train_config.flooding_level > 0:
                    train_step_metrics.update({"flood": flood.item()})
                self.log_manager.log_metrics(
                    metrics=train_step_metrics,
                    step=self.global_step,
                    epoch=self.epoch,
                    part="train",
                )
            pbar.update(n_samples)

    def run_one_step(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run one step (batch) of training.

        Parameters
        ----------
        input_tensors : dict
            the tensors to be processed for training one step (batch), with the following items:
                - "signals" (required): the input ECG signals
                - "chagas" (optional): the chagas classification labels

        Returns
        -------
        out_tensors : dict
            with the following items (some are optional):
            - "chagas": the chagas classification predictions, of shape ``(batch_size,)``.
            - "chagas_logits": the chagas classification logits, of shape ``(batch_size, n_classes)``.
            - "chagas_prob": the chagas classification probabilities, of shape ``(batch_size, n_classes)``.
            - "chagas_loss": the Dx classification loss

        """
        # input_tensors = {k: v.to(self.device) for k, v in input_tensors.items()}
        return self.model(input_tensors)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on the given data loader"""

        self.model.eval()

        all_outputs = []
        all_labels = []

        with tqdm(
            total=len(data_loader.dataset),
            desc="Evaluation",
            unit="signal",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
        ) as pbar:
            for idx, input_tensors in enumerate(data_loader):
                # input_tensors is assumed to be a dict of tensors, with the following items:
                # "signals" (required): the input image list
                # "chagas" (required): the chagas classification labels
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                labels = {"chagas": input_tensors.pop("chagas")}
                outputs = self.model.inference(input_tensors["signals"])
                outputs.drop(["chagas_logits", "chagas_loss"])  # reduce memory usage
                all_outputs.append(outputs)
                all_labels.append(labels)

                pbar.update(input_tensors["signals"].shape[self.batch_dim])

        eval_res = compute_challenge_metrics(all_labels, all_outputs)

        if self.val_train_loader is not None:
            log_head_num = 5
            head_scalar_preds = all_outputs[0].chagas_prob[:log_head_num]
            head_binary_preds = all_outputs[0].chagas[:log_head_num]
            head_labels = all_labels[0]["chagas"][:log_head_num]
            log_head_num = min(log_head_num, len(head_scalar_preds))
            for n in range(log_head_num):
                msg = textwrap.dedent(
                    f"""
                ----------------------------------------------
                Chagas scalar predictions:    {[round(item, 3) for item in head_scalar_preds[n].tolist()]}
                Chagas binary predictions:    {head_binary_preds[n]}
                Chagas labels:                {bool(head_labels[n])}
                ----------------------------------------------
                """
                )
                self.log_manager.log_message(msg)

        self.model.train()

        return eval_res

    @property
    def batch_dim(self) -> int:
        """
        batch dimension, usually 0,
        but can be 1 for some models, e.g. RR_LSTM
        """
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        return ["chagas_classes"]

    @property
    def save_prefix(self) -> str:
        prefix = self._model.__name__
        if hasattr(self._model.config, "cnn"):
            prefix = f"{prefix}_{self._model.config.cnn.name}_epoch"
        else:
            prefix = f"{prefix}_epoch"
        return prefix

    def extra_log_suffix(self) -> str:
        suffix = super().extra_log_suffix()
        if hasattr(self._model.config, "cnn"):
            suffix = f"{suffix}_{self._model.config.cnn.name}"
        return suffix

    def _setup_criterion(self) -> None:
        # since criterion is defined in the model,
        # override this method to do nothing
        pass

    def save_checkpoint(self, path: str) -> None:
        """Save the current state of the trainer to a checkpoint.

        Parameters
        ----------
        path : str
            Path to save the checkpoint

        """
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_config": make_safe_globals(self.model_config),
                "train_config": make_safe_globals(self.train_config),
                "epoch": self.epoch,
            },
            path,
        )


@torch.no_grad()
def run_chagas_model(
    chagas_model: CRNN_CINC2025, ds: CINC2025Dataset
) -> Tuple[List[CINC2025Outputs], List[Dict[str, torch.Tensor]]]:
    """Run the chagas classification model on the
    given data loader on different thresholds.

    Parameters
    ----------
    chagas_model : CRNN_CINC2025
        The chagas classification model to be run.
    ds : CINC2025Dataset
        The dataset for running the model.
    thresholds : Sequence[float]
        The thresholds for the evaluation.

    Returns
    -------
    outputs : list of CINC2025Outputs
        The outputs of the model on the dataset.
    labels : list of dict
        The labels of the dataset.

    """
    all_outputs = []
    all_labels = []
    data_loader = DataLoader(
        dataset=ds,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    with tqdm(
        total=len(data_loader.dataset),
        desc="Evaluation",
        unit="signal",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False,
    ) as pbar:
        for input_tensors in data_loader:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            labels = {"chagas": input_tensors.pop("chagas")}
            outputs = chagas_model.inference(input_tensors["signals"].to(chagas_model.device))
            outputs.drop(["chagas_logits", "chagas_loss"])  # reduce memory usage
            all_outputs.append(outputs)
            all_labels.append(labels)

            pbar.update(input_tensors["signals"].shape[0])

    return all_outputs, all_labels


@torch.no_grad()
def _evaluate_chagas_model(
    outputs: List[CINC2025Outputs], labels: List[Dict[str, torch.Tensor]], thresholds: Sequence[float]
) -> Dict[float, Dict[str, float]]:
    """Evaluate the chagas classification model on the
    given data loader on different thresholds.

    Parameters
    ----------
    outputs : list of CINC2025Outputs
        The outputs of the model on the dataset.
    labels : list of dict
        The labels of the dataset.
    thresholds : Sequence[float]
        The thresholds for the evaluation.

    Returns
    -------
    eval_res : dict
        The evaluation results on different thresholds.

    """
    eval_res = {}
    for threshold in thresholds:
        num_samples = 0
        num_positive = 0
        for output in outputs:
            output.chagas = (output.chagas_prob[:, 1] > threshold).tolist()
            output.chagas_threshold = threshold
            num_samples += len(output.chagas)
            num_positive += sum(output.chagas)
        # note that adjust the threshold for chagas binary classification
        # only affects accuracy, f_measure
        # but not auroc, auprc, challenge_score since they are computed based on the probability
        eval_res[threshold] = compute_challenge_metrics(labels=labels, outputs=outputs)
        eval_res[threshold].update({"positive_rate": num_positive / num_samples})
        print(f"threshold: {threshold}")
        print(f"metrics: {eval_res[threshold]}")
        print("-" * 80)
        print("\n")
    return eval_res


@torch.no_grad()
def evaluate_chagas_model(
    chagas_model: CRNN_CINC2025, ds: CINC2025Dataset, thresholds: Sequence[float]
) -> Dict[float, Dict[str, float]]:
    """Evaluate the chagas classification model on the
    given data loader on different thresholds.

    Parameters
    ----------
    chagas_model : CRNN_CINC2025
        The chagas classification model to be evaluated.
    ds : CINC2025Dataset
        The dataset for evaluation.
    thresholds : Sequence[float]
        The thresholds for the evaluation.

    Returns
    -------
    eval_res : dict
        The evaluation results on different thresholds.

    """
    all_outputs, all_labels = run_chagas_model(
        chagas_model=chagas_model,
        ds=ds,
        thresholds=thresholds,
    )
    eval_res = _evaluate_chagas_model(
        outputs=all_outputs,
        labels=all_labels,
        thresholds=thresholds,
    )
    return eval_res


def get_args(**kwargs: Any):
    """NOT checked,"""
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train the Model on CINC2025 database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=24,
        help="the batch size for training",
        dest="batch_size",
    )
    parser.add_argument(
        "--keep-checkpoint-max",
        type=int,
        default=10,
        help="maximum number of checkpoints to keep. If set 0, all checkpoints will be kept",
        dest="keep_checkpoint_max",
    )
    # parser.add_argument(
    #     "--optimizer", type=str, default="adam",
    #     help="training optimizer",
    #     dest="train_optimizer")
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="train with more debugging information",
        dest="debug",
    )

    args = vars(parser.parse_args())

    cfg.update(args)

    return CFG(cfg)


if __name__ == "__main__":
    # WARNING: most training were done in notebook,
    # NOT in cli
    train_config = get_args(**TrainCfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: adjust for CINC2025
    model_config = deepcopy(ModelCfg)
    # adjust the model configuration if necessary
    model = CRNN_CINC2025(config=model_config)

    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)
    model.to(device=device)

    trainer = CINC2025Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
        lazy=True,
    )

    try:
        best_model_state_dict = trainer.train()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
