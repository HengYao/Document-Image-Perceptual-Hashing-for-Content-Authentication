import PIL
import cv2
import numpy as np
import os
import re
import csv
import time
import pickle
import logging
from PIL import Image

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.utils
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage
from transformers import BertTokenizer

from ocr import ocr
from options import HiDDenConfiguration, TrainingOptions
from hidden import Hidden


def image_to_tensor(image):

    image_tensor = torch.Tensor(image)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor / 127.5 - 1
    return image_tensor


def tensor_to_image(tensor):

    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image + 1) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)

def single_image_to_tensor(image_path,device):
    image = Image.open(image_path)
    image_tensor = TF.to_tensor(image).to(device)
    image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
    image_tensor.unsqueeze_(0)

    return image_tensor

def save_images(original_images, watermarked_images, epoch, folder, resize_to=None):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    stacked_images = torch.cat([images, watermarked_images], dim=0)
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    torchvision.utils.save_image(stacked_images, filename, original_images.shape[0], normalize=False)


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file


def save_checkpoint(model: Hidden, experiment_name: str, epoch: int, checkpoint_folder: str):
    """ Saves a checkpoint at the end of an epoch. """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = f'{experiment_name}--epoch-{epoch}.pyt'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        'deephash-model': model.deephash.state_dict(),
        'deephash-optim': model.optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')


# def load_checkpoint(hidden_net: Hidden, options: Options, this_run_folder: str):
def load_last_checkpoint(checkpoint_folder):
    """ Load the last checkpoint from the given folder """
    last_checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
    checkpoint = torch.load(last_checkpoint_file)

    return checkpoint, last_checkpoint_file


def model_from_checkpoint(hidden_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    hidden_net.deephash.load_state_dict(checkpoint['deephash-model'])
    hidden_net.optimizer.load_state_dict(checkpoint['deephash-optim'])


def load_options(options_file_name):
    """ Loads the training, model, and noise configurations from the given folder """
    with open(os.path.join(options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        hidden_config = pickle.load(f)

    return train_options, hidden_config


def process_image_and_text(image):

    image = np.array(image)
    ocr_result, _ = ocr(image)
    all_text = ' '.join([ocr_result[key][1] for key in ocr_result])

    tokenizer = BertTokenizer.from_pretrained('.../BERT/vocab.txt')
    texts = '[CLS] ' + all_text + ' [SEP]'

    tokens, segments, input_masks = [], [], []


    tokenized_text = tokenizer.tokenize(texts)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens.append(indexed_tokens)
    segments.append([0] * len(indexed_tokens))
    input_masks.append([1] * len(indexed_tokens))

    max_len = 256

    for j in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        input_masks[j] += padding

    tokens_tensor = torch.tensor(tokens)
    tokens_tensor = tokens_tensor.squeeze(0)
    segments_tensors = torch.tensor(segments)
    segments_tensors = segments_tensors.squeeze(0)
    input_masks_tensors = torch.tensor(input_masks)
    input_masks_tensors = input_masks_tensors.squeeze(0)

    return tokens_tensor, segments_tensors, input_masks_tensors,image


class CustomDataset(Dataset):
    def __init__(self, image_folder, transform):
        self.image_folder = image_folder
        self.transform = transform
        self.image_list =[os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.endswith('.jpg')]


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_folder = self.image_list[idx]
        image = Image.open(image_folder).convert('RGB')
        image_tensor = self.transform(image)
        tokens_tensor, segments_tensor, input_masks_tensor, image = process_image_and_text(image)

        return tokens_tensor, segments_tensor, input_masks_tensor, image_tensor

def custom_collate_fn(batch):
    tokens_tensor, segments_tensor, input_masks_tensor, image_tensor = zip(*batch)
    tokens_tensor = torch.stack(tokens_tensor)
    segments_tensor = torch.stack(segments_tensor)
    input_masks_tensor = torch.stack(input_masks_tensor)
    image_tensor = torch.stack(image_tensor)


    return tokens_tensor, segments_tensor, input_masks_tensor, image_tensor


def get_data_loaders(hidden_config: HiDDenConfiguration, train_options: TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([transforms.Resize((hidden_config.H,hidden_config.W)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]),
        'val': transforms.Compose([transforms.Resize((hidden_config.H,hidden_config.W)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
    }

    train_images = CustomDataset(train_options.train_folder, data_transforms['train'])
    train_loader = DataLoader(train_images, batch_size=train_options.batch_size, shuffle=False,
                                               num_workers=0,collate_fn=custom_collate_fn)

    validation_images = CustomDataset(train_options.validation_folder, data_transforms['val'])
    validation_loader = DataLoader(validation_images, batch_size=train_options.batch_size,
                                                    shuffle=False, num_workers=0,collate_fn=custom_collate_fn)

    return train_loader, validation_loader

def get_testdata_loaders(path,batch_size):

    data_transforms = {
        'train': transforms.Compose([transforms.Resize((256,256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]),
        'val': transforms.Compose([transforms.Resize((256,256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
    }

    test_images = CustomDataset(path, data_transforms['val'])
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=custom_collate_fn)


    return  test_loader

def log_progress(losses_accu):
    log_print_helper(losses_accu, logging.info)


def print_progress(losses_accu):
    log_print_helper(losses_accu, print)


def log_print_helper(losses_accu, log_or_print_func):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        log_or_print_func(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))


def create_folder_for_run(runs_folder, experiment_name):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{experiment_name}-{time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))

    return this_run_folder

def create_folder_for_test(runs_folder):

    this_test_folder = os.path.join( f'{runs_folder}-{time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_test_folder)

    return this_test_folder

def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration(sec)']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss) for loss in losses_accu.values()] + [
            '{:.0f}'.format(duration)]
        writer.writerow(row_to_write)

def write_losses_validation(file_name, losses, epoch):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        row_to_write = [epoch] + ['{:.5f}'.format(loss) for loss in losses]
        writer.writerow(row_to_write)

def write_losses_test(file_name, losses):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        row_to_write = ['{:.5f}'.format(loss) for loss in losses]
        writer.writerow(row_to_write)

def write_hash(file_name, hash, epoch, train_options):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(train_options.batch_size):
            row_to_write = [epoch] + ['{:.5f}'.format(single_code.item()) for single_code in hash[i]]
            writer.writerow(row_to_write)

def write_hash_test(file_name, hash, batch_size):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(batch_size):
            row_to_write = ['{:.5f}'.format(single_code.item()) for single_code in hash[i]]
            writer.writerow(row_to_write)

def save_images_all(original_images, watermarked_images,noised_images, epoch, folder, resize_to=None):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()
    noised_images = noised_images[:noised_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2
    noised_images = (noised_images + 1) / 2
    noised_images[noised_images > 1] = 1
    noised_images[noised_images < 0] = 0

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)
        noised_images = F.interpolate(noised_images, size=resize_to)

    stacked_images = torch.cat([images, watermarked_images, noised_images], dim=0)
    filename = os.path.join(folder, 'step-{}.png'.format(epoch))
    torchvision.utils.save_image(stacked_images, filename, original_images.shape[0], normalize=False)

