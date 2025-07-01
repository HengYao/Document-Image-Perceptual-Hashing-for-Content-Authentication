import os
import pprint
import argparse
import torch
import pickle
import utils
import logging
import sys

from options import *
from hidden import Hidden

from train import train


def main():
    device = torch.device("cuda")

    parent_parser = argparse.ArgumentParser(description='Training of Deephash nets')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')
    new_run_parser = subparsers.add_parser('new', help='starts a new run')  # new or continue
    # 训练图像的位置
    new_run_parser.add_argument('--data-dir', '-d', default=r'...', type=str,
                                help='The directory where the data is stored.')
    #
    new_run_parser.add_argument('--batch-size', '-b', default=71, type=int, help='The batch size.')
    new_run_parser.add_argument('--epochs', '-e', default=100, type=int, help='Number of epochs to run the simulation.')
    new_run_parser.add_argument('--name', default='FusionHash', type=str, help='The name of the experiment.')
    new_run_parser.add_argument('--size', '-s', default=256, type=int,
                                help='The size of the images (images are square so this is height and width).')
    new_run_parser.add_argument('--hash', '-l', default=50, type=int, help='Length of hash code.')
    new_run_parser.add_argument('--continue-from-folder', '-c', default='continue', type=str,
                                help='The folder from where to continue a previous run. Leave blank if you are starting a new experiment.')


    continue_parser = subparsers.add_parser('continue', help='Continue a previous run')
    continue_parser.add_argument('--folder', '-f', default=r'...', type=str,
                                 help='Continue from the last checkpoint in this folder.')
    continue_parser.add_argument('--data-dir', '-d', default=r'...', type=str,
                                 help='The directory where the data is stored. Specify a value only if you want to override the previous value.')
    continue_parser.add_argument('--epochs', '-e', required=False, type=int,
                                help='Number of epochs to run the simulation. Specify a value only if you want to override the previous value.')


    args = parent_parser.parse_args()
    checkpoint = None
    loaded_checkpoint_file_name = None

    if args.command == 'continue':
        this_run_folder = args.folder
        options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
        train_options, hidden_config = utils.load_options(options_file)
        checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(os.path.join(this_run_folder, 'checkpoints'))
        train_options.start_epoch = checkpoint['epoch'] + 1
        if args.data_dir is not None:
            train_options.train_folder = os.path.join(args.data_dir, 'train')
            train_options.validation_folder = os.path.join(args.data_dir, 'val')
        if args.epochs is not None:
            if train_options.start_epoch < args.epochs:
                train_options.number_of_epochs = args.epochs
            else:
                print(f'Command-line specifies of number of epochs = {args.epochs}, but folder={args.folder} '
                      f'already contains checkpoint for epoch = {train_options.start_epoch}.')
                exit(1)

    else:
        assert args.command == 'new'
        start_epoch = 1

        train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            train_folder=os.path.join(args.data_dir, 'train'),
            validation_folder=os.path.join(args.data_dir, 'val'),
            runs_folder=os.path.join('.', 'runs'),
            start_epoch=start_epoch,
            experiment_name=args.name)


        hidden_config = HiDDenConfiguration(H=args.size, W=args.size, L=args.hash)
        this_run_folder = utils.create_folder_for_run(train_options.runs_folder, args.name)
        with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(train_options, f,0)
            pickle.dump(hidden_config, f,0)


    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(this_run_folder, f'{train_options.experiment_name}.log')),
                            logging.StreamHandler(sys.stdout)
                        ])

    model = Hidden(configuration=hidden_config, device=device)

    if args.command == 'continue':
        # if we are continuing, we have to load the model params
        assert checkpoint is not None
        logging.info(f'Loading checkpoint from file {loaded_checkpoint_file_name}')
        utils.model_from_checkpoint(model, checkpoint)

    logging.info('Deep hash model: {}\n'.format(model.to_stirng()))
    logging.info('Model Configuration:\n')
    logging.info(pprint.pformat(vars(hidden_config)))
    logging.info('\nTraining train_options:\n')
    logging.info(pprint.pformat(vars(train_options)))

    train(model, device, hidden_config, train_options, this_run_folder)

if __name__ == '__main__':
    main()

