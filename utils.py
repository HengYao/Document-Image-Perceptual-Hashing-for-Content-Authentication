import csv
import hashlib
import logging
import os
import pickle
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from transformers import BertTokenizer

from ocr import ocr
from options import HiDDenConfiguration, TrainingOptions


PROJECT_ROOT = Path(__file__).resolve().parent
BERT_MODEL_DIR = PROJECT_ROOT / "weights" / "chinese_bert"


def save_checkpoint(
    model,
    experiment_name,
    epoch,
    checkpoint_folder,
    validation_loss,
    best_validation_loss,
    is_best,
):
    checkpoint_folder = Path(checkpoint_folder)
    epoch_folder = checkpoint_folder / "epochs"
    epoch_folder.mkdir(parents=True, exist_ok=True)

    epoch_weights_path = epoch_folder / (
        f"{experiment_name}--epoch-{epoch:03d}.weights.pt"
    )
    model_state = {
        "model": model.deephash.state_dict(),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "hash_length": model.config.L,
    }
    torch.save(model_state, epoch_weights_path)

    resume_state = {
        "deephash-model": model.deephash.state_dict(),
        "deephash-optim": model.optimizer.state_dict(),
        "amp-scaler": model.scaler.state_dict(),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "hash_length": model.config.L,
    }
    last_path = checkpoint_folder / "last.checkpoint.pt"
    temporary_last_path = checkpoint_folder / "last.checkpoint.tmp"
    torch.save(resume_state, temporary_last_path)
    os.replace(temporary_last_path, last_path)

    if is_best:
        best_path = checkpoint_folder / f"{experiment_name}--best.weights.pt"
        shutil.copy2(epoch_weights_path, best_path)
        logging.info("New best checkpoint: %s", best_path)

    logging.info("Saved epoch checkpoint: %s", epoch_weights_path)


def load_checkpoint(checkpoint_file, device):
    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )
    return checkpoint, str(checkpoint_file)


def last_checkpoint_from_folder(folder):
    last_path = Path(folder) / "last.checkpoint.pt"
    if not last_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {last_path}")
    return last_path


def load_last_checkpoint(checkpoint_folder, device):
    checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
    return load_checkpoint(checkpoint_file, device)


def model_from_checkpoint(hidden_net, checkpoint):
    hidden_net.deephash.load_state_dict(checkpoint["deephash-model"])
    hidden_net.optimizer.load_state_dict(checkpoint["deephash-optim"])
    if "amp-scaler" in checkpoint:
        hidden_net.scaler.load_state_dict(checkpoint["amp-scaler"])


def load_options(options_file_name):
    with open(options_file_name, "rb") as handle:
        train_options = pickle.load(handle)
        hidden_config = pickle.load(handle)
    return train_options, hidden_config


class CustomDataset(Dataset):
    def __init__(self, image_folder, transform, token_cache_folder):
        self.image_folder = Path(image_folder)
        self.transform = transform
        self.token_cache_folder = Path(token_cache_folder)
        self.token_cache_folder.mkdir(parents=True, exist_ok=True)
        self.image_list = sorted(self.image_folder.glob("*.jpg"))
        self.tokenizer = BertTokenizer.from_pretrained(str(BERT_MODEL_DIR))

        if not self.image_list:
            raise FileNotFoundError(
                f"No directly contained JPG files found in {self.image_folder}"
            )

    def __len__(self):
        return len(self.image_list)

    def _cache_path(self, image_path):
        digest = hashlib.sha1(
            str(image_path.resolve()).encode("utf-8")
        ).hexdigest()
        return self.token_cache_folder / f"{digest}.npz"

    def _tokens_for_image(self, image_path, image):
        cache_path = self._cache_path(image_path)
        if cache_path.exists():
            with np.load(cache_path) as cached:
                return (
                    torch.from_numpy(cached["input_ids"].copy()),
                    torch.from_numpy(cached["token_type_ids"].copy()),
                    torch.from_numpy(cached["attention_mask"].copy()),
                )

        ocr_result, _ = ocr(np.asarray(image))
        all_text = " ".join(
            ocr_result[key][1] for key in sorted(ocr_result)
        )
        encoded = self.tokenizer(
            all_text,
            add_special_tokens=True,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        arrays = {
            "input_ids": encoded["input_ids"][0].astype(np.int64),
            "token_type_ids": encoded["token_type_ids"][0].astype(np.int64),
            "attention_mask": encoded["attention_mask"][0].astype(np.int64),
        }
        temporary_path = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, cache_path)
        return tuple(torch.from_numpy(value) for value in arrays.values())

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        tokens, segments, input_masks = self._tokens_for_image(
            image_path,
            image,
        )
        return tokens, segments, input_masks, image_tensor


class GroupBatchSampler(Sampler):
    def __init__(self, dataset_size, group_size, shuffle):
        if group_size < 3:
            raise ValueError("Group size must be at least 3")
        if dataset_size % group_size != 0:
            raise ValueError(
                f"{dataset_size} images cannot form complete groups of "
                f"{group_size}"
            )
        self.dataset_size = dataset_size
        self.group_size = group_size
        self.group_count = dataset_size // group_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            group_order = torch.randperm(self.group_count).tolist()
        else:
            group_order = range(self.group_count)
        for group_index in group_order:
            start = group_index * self.group_size
            yield list(range(start, start + self.group_size))

    def __len__(self):
        return self.group_count


def get_data_loaders(
    hidden_config: HiDDenConfiguration,
    train_options: TrainingOptions,
):
    transform = transforms.Compose(
        [
            transforms.Resize((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )
    cache_root = Path(train_options.token_cache_folder)
    train_dataset = CustomDataset(
        train_options.train_folder,
        transform,
        cache_root / "train",
    )
    validation_dataset = CustomDataset(
        train_options.validation_folder,
        transform,
        cache_root / "validation",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=GroupBatchSampler(
            len(train_dataset),
            train_options.batch_size,
            shuffle=True,
        ),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=GroupBatchSampler(
            len(validation_dataset),
            train_options.batch_size,
            shuffle=False,
        ),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader


def create_folder_for_run(runs_folder, experiment_name):
    runs_folder = Path(runs_folder)
    runs_folder.mkdir(parents=True, exist_ok=True)
    run_folder = runs_folder / (
        f"{experiment_name}-{time.strftime('%Y.%m.%d--%H-%M-%S')}"
    )
    run_folder.mkdir()
    (run_folder / "checkpoints").mkdir()
    return str(run_folder)


def write_metrics(file_name, metrics, epoch, duration=None):
    file_path = Path(file_name)
    new_file = not file_path.exists()
    fieldnames = ["epoch", *metrics.keys()]
    if duration is not None:
        fieldnames.append("duration_sec")

    with file_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        row = {"epoch": epoch, **metrics}
        if duration is not None:
            row["duration_sec"] = round(duration, 2)
        writer.writerow(row)
