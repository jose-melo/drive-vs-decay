import os, glob
from enum import Enum

from tabulate import tabulate
import torch
from time import sleep

MODEL_CP_NAME = "epoch_{epoch}.pth"

class EarlyStopSignal(Enum):
    CONTINUE = 0
    STOP = 1
    END = 2

class EarlyStopCounter:
    def __init__(self, args, jobname, dataset_name, device=None, is_distributed=False):
        self._jobname = jobname
        self.checkpoint_dir = os.path.join("./checkpoints", dataset_name, self._jobname)
        self.last_model_path = None
        self.is_distributed = is_distributed

        self.seed = args.np_seed

        self.num_inc_train_loss_epochs = 0

        self.num_train_improvements_since_cache = 0

        self.num_val_improvements_since_cache = 0
        self.val_cache_cadence = args.exp_val_cache_cadence

        self.cadence_type = args.exp_cadence_type
        err_mess = "Chosen cadence type not allowed"
        assert self.cadence_type in ["improvement", "recurrent", "None"], err_mess

        self.iter_recurrent_caching = 0

        if args.exp_cache_cadence == -1:
            self.cache_cadence = float("inf")
        else:
            self.cache_cadence = args.exp_cache_cadence

        self.min_train_loss = float("inf")

        self.best_val_score = float("-inf")

        self.patience = args.exp_patience
        self.args = args

        self.early_stop_signal_message = (
            f"Training loss has not improved "
            f"for {self.patience} contiguous epochs. "
            f"Stopping training now."
        )

        self.training_over_stop_signal_message = (
            f"Training completed."    f"Moving to final evaluation now."
        )

        self.device = device
        if not self.is_distributed:
            self.main_process = True
        else:
            self.main_process = self.device == 0
        if self.main_process:
            if not self.args.load_from_checkpoint:
                self.clear_checkpoint_path()

    def update(
        self,
        train_loss,
        context_encoder,
        target_encoder,
        predictor,
        optimizer,
        scaler,
        scheduler,
        weightdecay_scheduler,
        epoch,
        end_experiment,
        val_score=None,
    ):

        if self.cadence_type == "improvement":
            return self.update_improvement(
                train_loss,
                context_encoder,
                target_encoder,
                predictor,
                optimizer,
                scaler,
                scheduler,
                weightdecay_scheduler,
                epoch,
                end_experiment,
                val_score,
            )
        elif self.cadence_type == "recurrent":
            return self.update_recurrent(
                train_loss,
                context_encoder,
                target_encoder,
                predictor,
                optimizer,
                scaler,
                scheduler,
                weightdecay_scheduler,
                epoch,
                end_experiment,
            )

        else:
            return

    def update_recurrent(
        self,
        train_loss,
        context_encoder,
        target_encoder,
        predictor,
        optimizer,
        scaler,
        scheduler,
        weightdecay_scheduler,
        epoch,
        end_experiment,
    ):

        self.iter_recurrent_caching += 1
        if self.iter_recurrent_caching > self.cache_cadence or end_experiment:
            if self.iter_recurrent_caching > self.cache_cadence:
                pass
            else:
                pass
            self.cache_model(
                context_encoder,
                target_encoder,
                predictor,
                optimizer,
                scaler,
                scheduler,
                weightdecay_scheduler,
                epoch,
                train_loss,
            )
            self.best_epoch = epoch
            self.iter_recurrent_caching = 0

        if end_experiment:
            return (
                EarlyStopSignal.END,
                context_encoder,
                target_encoder,
                predictor,
                optimizer,
                scaler,
                scheduler,
                weightdecay_scheduler,
            )
        else:
            return (
                EarlyStopSignal.CONTINUE,
                context_encoder,
                target_encoder,
                predictor,
                optimizer,
                scaler,
                scheduler,
                weightdecay_scheduler,
            )

    def update_improvement(
        self,
        train_loss,
        context_encoder,
        target_encoder,
        predictor,
        optimizer,
        scaler,
        scheduler,
        weightdecay_scheduler,
        epoch,
        end_experiment,
        val_score,
    ):

        if val_score is not None and val_score > self.best_val_score:
            self.best_val_score = val_score
            self.num_val_improvements_since_cache += 1

            if self.main_process and (
                self.num_val_improvements_since_cache >= self.val_cache_cadence
            ):
                self.cache_model(
                    context_encoder,
                    target_encoder,
                    predictor,
                    optimizer,
                    scaler,
                    scheduler,
                    weightdecay_scheduler,
                    epoch,
                    train_loss,
                )
                self.num_val_improvements_since_cache = 0

        if train_loss < self.min_train_loss:
            self.min_train_loss = train_loss
            self.num_inc_train_loss_epochs = 0
            self.num_train_improvements_since_cache += 1

            if self.main_process and (
                self.num_train_improvements_since_cache >= self.cache_cadence
            ):
                self.cache_model(
                    context_encoder,
                    target_encoder,
                    predictor,
                    optimizer,
                    scaler,
                    scheduler,
                    weightdecay_scheduler,
                    epoch,
                    train_loss,
                )
                self.num_train_improvements_since_cache = 0
            self.best_epoch = epoch
        else:
            self.num_inc_train_loss_epochs += 1

        if end_experiment:
            load_pth = os.path.join(
                self.checkpoint_dir, MODEL_CP_NAME.format(epoch=self.best_epoch)
            )
            try:
                return (
                    EarlyStopSignal.END,
                    *self.load_model(
                        load_pth,
                        context_encoder,
                        target_encoder,
                        predictor,
                        optimizer,
                        scaler,
                        scheduler,
                        weightdecay_scheduler,
                    ),
                )
            except Exception as e:
                return (
                    EarlyStopSignal.END,
                    context_encoder,
                    target_encoder,
                    predictor,
                    optimizer,
                    scaler,
                    scheduler,
                    weightdecay_scheduler,
                )

        elif self.patience == -1:
            return (
                EarlyStopSignal.CONTINUE,
                context_encoder,
                target_encoder,
                predictor,
                optimizer,
                scaler,
                scheduler,
                weightdecay_scheduler,
            )

        elif self.num_inc_train_loss_epochs > self.patience:
            load_pth = os.path.join(
                self.checkpoint_dir, MODEL_CP_NAME.format(epoch=self.best_epoch)
            )
            try:
                return (
                    EarlyStopSignal.STOP,
                    *self.load_model(
                        load_pth,
                        context_encoder,
                        target_encoder,
                        predictor,
                        optimizer,
                        scaler,
                        scheduler,
                        weightdecay_scheduler,
                    ),
                )
            except Exception:
                return (
                    EarlyStopSignal.STOP,
                    context_encoder,
                    target_encoder,
                    predictor,
                    optimizer,
                    scaler,
                    scheduler,
                    weightdecay_scheduler,
                )

        return (
            EarlyStopSignal.CONTINUE,
            context_encoder,
            target_encoder,
            predictor,
            optimizer,
            scaler,
            scheduler,
            weightdecay_scheduler,
        )

    def clear_checkpoint_path(self):
        name_list = glob.glob(os.path.join(self.checkpoint_dir, "epoch_*.pth"))
        if len(name_list) > 0:
            for f in name_list:
                os.remove(f)

    def cache_model(
        self,
        context_encoder,
        target_encoder,
        predictor,
        optimizer,
        scaler,
        scheduler,
        weightdecay_scheduler,
        epoch,
        train_loss,
        save_pth=None,
    ):

        checkpoint_dict = {
            "epoch": epoch,
            "context_encoder": context_encoder.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "predictor": predictor.state_dict_(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "weightdecay": weightdecay_scheduler.state_dict(),
            "train_loss": train_loss,
        }
        if save_pth is None:
            self.last_model_path = os.path.join(
                self.checkpoint_dir, MODEL_CP_NAME.format(epoch=epoch)
            )
        else:
            self.last_model_path = os.path.join(
                save_pth, MODEL_CP_NAME.format(epoch=epoch)
            )

        model_is_checkpointed = False
        counter = 0
        while model_is_checkpointed is False and counter < 10000:
            if counter % 10 == 0:

                pass
            torch.save(checkpoint_dict, self.last_model_path)

            if os.path.isfile(self.last_model_path):
                model_is_checkpointed = True

            if model_is_checkpointed is False:
                sleep(0.5)

            counter += 1

    def load_model(
        self,
        load_pth,
        context_encoder,
        target_encoder,
        predictor,
        optimizer,
        scaler,
        scheduler,
        weightdecay_scheduler,
    ):

        state_dict = torch.load(load_pth, map_location=self.device)
        context_encoder.load_state_dict(state_dict["context_encoder"])
        target_encoder.load_state_dict(state_dict["target_encoder"])

        predictor.load_state_dict_(state_dict["predictor"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scaler.load_state_dict(state_dict["scaler"])
        scheduler.load_state_dict(state_dict["scheduler"])
        weightdecay_scheduler.load_state_dict(state_dict["weightdecay"])
        start_epoch = state_dict["epoch"]
        train_loss = state_dict["train_loss"]

        configs = {
            "learning_rate": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0]["weight_decay"],
            "start_epoch": start_epoch,
            "train_loss": train_loss,
        }
        if isinstance(start_epoch, torch.Tensor):
            start_epoch = start_epoch.cpu().numpy()

        return (
            context_encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            scheduler,
            weightdecay_scheduler,
            start_epoch,
        )

    def get_job_name(self):
        return self._jobname
