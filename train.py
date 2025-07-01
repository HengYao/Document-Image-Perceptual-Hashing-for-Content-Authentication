import os
import time
import torch
import numpy as np
from transformers import BertTokenizer

import utils
import logging
from collections import defaultdict

from options import *
from hidden import Hidden


def train(model: Hidden,
          device: torch.device,
          hidden_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str):
    """
    Trains the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param hidden_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger:
    :return:TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    """

    train_data, val_data = utils.get_data_loaders(hidden_config, train_options)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 5

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))

        epoch_start = time.time()
        step = 1
        for tokens_tensor,segments_tensor,input_masks_tensor,image in train_data:
            image = image.to(device)
            tokens = tokens_tensor.to(device)
            segments = segments_tensor.to(device)
            input_masks = input_masks_tensor.to(device)
            training_losses, hash = model.train_on_batch(tokens, segments, input_masks,image)

            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                logging.info('similar loss: {}'.format(training_losses['similar_loss    ']))
                logging.info('differe loss: {}'.format(training_losses['different_loss  ']))
                logging.info('    loss    : {}'.format(training_losses['loss            ']))
                logging.info('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)

        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))

        for tokens_tensor,segments_tensor,input_masks_tensor,image in val_data:
            image = image.to(device)
            tokens = tokens_tensor.to(device)
            segments = segments_tensor.to(device)
            input_masks = input_masks_tensor.to(device)
            validation_losses, hash = model.validate_on_batch(tokens, segments, input_masks,image)
            utils.write_losses_validation(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch)

        utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
