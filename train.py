import logging
import os
import time

import torch

import utils
from hidden import Hidden
from options import HiDDenConfiguration, TrainingOptions


METRIC_NAMES = ("similar_loss", "different_loss", "loss")


def _empty_totals():
    return {name: 0.0 for name in METRIC_NAMES}


def _averages(totals, sample_count):
    if sample_count == 0:
        raise RuntimeError("Data loader produced no usable batches")
    return {
        name: totals[name] / sample_count
        for name in METRIC_NAMES
    }


def _move_batch(batch, device):
    tokens, segments, input_masks, images = batch
    return (
        tokens.to(device, non_blocking=True),
        segments.to(device, non_blocking=True),
        input_masks.to(device, non_blocking=True),
        images.to(device, non_blocking=True),
    )


def train(
    model: Hidden,
    device: torch.device,
    hidden_config: HiDDenConfiguration,
    train_options: TrainingOptions,
    this_run_folder: str,
    best_validation_loss=float("inf"),
):
    train_data, val_data = utils.get_data_loaders(
        hidden_config,
        train_options,
    )
    patience = getattr(
        train_options,
        "early_stopping_patience",
        12,
    )
    min_delta = getattr(
        train_options,
        "early_stopping_min_delta",
        0.0001,
    )
    epochs_without_improvement = 0

    for epoch in range(
        train_options.start_epoch,
        train_options.number_of_epochs + 1,
    ):
        epoch_start = time.time()
        model.deephash.train()
        train_totals = _empty_totals()
        train_samples = 0

        logging.info(
            "Starting epoch %d/%d; batches=%d",
            epoch,
            train_options.number_of_epochs,
            len(train_data),
        )

        for step, batch in enumerate(train_data, start=1):
            tokens, segments, input_masks, images = _move_batch(
                batch,
                device,
            )
            batch_size = images.size(0)
            batch_losses, _ = model.train_on_batch(
                tokens,
                segments,
                input_masks,
                images,
            )
            for name in METRIC_NAMES:
                train_totals[name] += batch_losses[name] * batch_size
            train_samples += batch_size

            if step % 5 == 0 or step == len(train_data):
                logging.info(
                    "Epoch %d step %d/%d loss=%.6f",
                    epoch,
                    step,
                    len(train_data),
                    batch_losses["loss"],
                )

        train_metrics = _averages(train_totals, train_samples)
        train_duration = time.time() - epoch_start
        utils.write_metrics(
            os.path.join(this_run_folder, "train.csv"),
            train_metrics,
            epoch,
            train_duration,
        )

        model.deephash.eval()
        validation_totals = _empty_totals()
        validation_samples = 0

        with torch.no_grad():
            for batch in val_data:
                tokens, segments, input_masks, images = _move_batch(
                    batch,
                    device,
                )
                batch_size = images.size(0)
                batch_losses, _ = model.validate_on_batch(
                    tokens,
                    segments,
                    input_masks,
                    images,
                )
                for name in METRIC_NAMES:
                    validation_totals[name] += (
                        batch_losses[name] * batch_size
                    )
                validation_samples += batch_size

        validation_metrics = _averages(
            validation_totals,
            validation_samples,
        )
        utils.write_metrics(
            os.path.join(this_run_folder, "validation.csv"),
            validation_metrics,
            epoch,
        )

        current_validation_loss = validation_metrics["loss"]
        previous_best = best_validation_loss
        is_best = current_validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = current_validation_loss
        if current_validation_loss < previous_best - min_delta:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        utils.save_checkpoint(
            model=model,
            experiment_name=train_options.experiment_name,
            epoch=epoch,
            checkpoint_folder=os.path.join(
                this_run_folder,
                "checkpoints",
            ),
            validation_loss=current_validation_loss,
            best_validation_loss=best_validation_loss,
            is_best=is_best,
        )
        logging.info(
            "Epoch %d complete: train_loss=%.6f val_loss=%.6f best=%.6f duration=%.1fs",
            epoch,
            train_metrics["loss"],
            current_validation_loss,
            best_validation_loss,
            train_duration,
        )
        if patience > 0 and epochs_without_improvement >= patience:
            logging.info(
                "Early stopping after %d epochs without validation "
                "improvement greater than %.6f",
                epochs_without_improvement,
                min_delta,
            )
            break
