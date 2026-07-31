class TrainingOptions:
    def __init__(
        self,
        batch_size,
        number_of_epochs,
        train_folder,
        validation_folder,
        runs_folder,
        start_epoch,
        experiment_name,
        token_cache_folder,
        early_stopping_patience=12,
        early_stopping_min_delta=0.0001,
    ):
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.runs_folder = runs_folder
        self.start_epoch = start_epoch
        self.experiment_name = experiment_name
        self.token_cache_folder = token_cache_folder
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta


class HiDDenConfiguration:
    def __init__(self, H, W, L):
        self.H = H
        self.W = W
        self.L = L
