import argparse
import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import BertTokenizer

from imagetext import FusionModel
from options import HiDDenConfiguration


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_MODEL = ROOT / "weights" / "model_best.pt"
DEFAULT_CACHE = ROOT / "cache" / "tokens"


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def collect_images(folder, recursive):
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise NotADirectoryError("The image folder does not exist")
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    images = sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_key,
    )
    if not images:
        raise FileNotFoundError("No supported images were found")
    return images


class ImageDataset(Dataset):
    def __init__(self, images, image_size, cache_folder):
        self.images = images
        self.cache_folder = Path(cache_folder)
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self.tokenizer = BertTokenizer.from_pretrained(
            str(ROOT / "weights" / "chinese_bert")
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.images)

    def cache_path(self, path):
        stat = path.stat()
        identity = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()
        return self.cache_folder / f"{digest}.npz"

    def get_tokens(self, path, image):
        cache_path = self.cache_path(path)
        if cache_path.exists():
            with np.load(cache_path) as cached:
                return (
                    torch.from_numpy(cached["input_ids"].copy()),
                    torch.from_numpy(cached["token_type_ids"].copy()),
                    torch.from_numpy(cached["attention_mask"].copy()),
                )
        from ocr import ocr
        result, _ = ocr(np.asarray(image))
        text = " ".join(result[key][1] for key in sorted(result))
        encoded = self.tokenizer(
            text,
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
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(cache_path)
        return tuple(torch.from_numpy(value) for value in arrays.values())

    def __getitem__(self, index):
        path = self.images[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
            image_tensor = self.transform(image)
            tokens, segments, masks = self.get_tokens(path, image)
        return tokens, segments, masks, image_tensor


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return device


def load_model(model_file, device):
    checkpoint = torch.load(model_file, map_location=device, weights_only=True)
    state = checkpoint.get("model", checkpoint.get("deephash-model"))
    if state is None:
        raise KeyError("The model state is missing from the checkpoint")
    hash_length = int(
        checkpoint.get(
            "hash_length",
            state["fuse_fc.weight"].shape[0],
        )
    )
    config = HiDDenConfiguration(256, 256, hash_length)
    model = FusionModel(config, n_class=hash_length).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config, checkpoint.get("epoch")


def encode_images(model, loader, device, image_count):
    batches = []
    completed = 0
    with torch.no_grad():
        for tokens, segments, masks, images in loader:
            hashes = model(
                tokens.to(device, non_blocking=True),
                segments.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
                images.to(device, non_blocking=True),
            )
            hashes = hashes.detach().float().cpu()
            batches.append(hashes)
            completed += hashes.size(0)
            print(f"Encoded {completed}/{image_count}", flush=True)
    return torch.cat(batches, dim=0)


def write_group_csv(path, hashes, group_size):
    if group_size < 2:
        raise ValueError("Group size must be at least 2")
    if hashes.size(0) % group_size:
        raise ValueError("The image count cannot form complete groups")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    group_count = hashes.size(0) // group_size
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        for index in range(group_count):
            group = hashes[index * group_size:(index + 1) * group_size]
            distances = torch.mean((group[1:] - group[0]) ** 2, dim=1)
            writer.writerow([format(float(value), ".12g") for value in distances])
            print(f"Wrote group {index + 1}/{group_count}", flush=True)
    return group_count, group_size - 1


def write_all_csv(path, hashes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_count = hashes.size(0)
    if image_count < 2:
        raise ValueError("At least two images are required")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        for index in range(image_count):
            distances = torch.mean((hashes - hashes[index]) ** 2, dim=1)
            distances = torch.cat((distances[:index], distances[index + 1:]))
            writer.writerow([format(float(value), ".12g") for value in distances])
            if (index + 1) % 25 == 0 or index + 1 == image_count:
                print(f"Wrote row {index + 1}/{image_count}", flush=True)
    return image_count, image_count - 1


def add_common_arguments(parser, output_name):
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / output_name),
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    group = subparsers.add_parser("group")
    add_common_arguments(group, "group_mse.csv")
    group.add_argument("--group-size", required=True, type=int)
    all_pairs = subparsers.add_parser("all")
    add_common_arguments(all_pairs, "all_mse.csv")
    return parser


def main():
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    images = collect_images(args.source, args.recursive)
    device = choose_device(args.device)
    model, config, epoch = load_model(args.model, device)
    dataset = ImageDataset(images, config.H, args.cache)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    hashes = encode_images(model, loader, device, len(images))
    if args.mode == "group":
        rows, values = write_group_csv(args.output, hashes, args.group_size)
    else:
        rows, values = write_all_csv(args.output, hashes)
    print(f"Device: {device}")
    print(f"Checkpoint epoch: {epoch}")
    print(f"Images: {len(images)}")
    print(f"CSV rows: {rows}")
    print(f"MSE values per row: {values}")
    print(f"Output: {Path(args.output)}")


if __name__ == "__main__":
    main()
