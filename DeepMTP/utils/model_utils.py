import numpy as np

class EarlyStopping:
    '''Early stops the training if validation loss doesn't improve after a given patience.'''

    def __init__(self, use_early_stopping, patience=7, delta=0, metric_to_track='loss', verbose=False):
        '''
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        '''
        self.use_early_stopping = use_early_stopping
        self.patience = patience
        self.delta = delta
        self.metric_to_track = metric_to_track
        self.verbose = verbose

        self.early_stop_flag = False
        self.counter = 0

        self.best_score = None
        self.best_epoch = None
        self.best_model = None
        self.best_optimizer_state_dict = None
        self.best_performance_results = None

        if True in [
            m in self.metric_to_track
            for m in [
                'auroc',
                'aupr',
                'recall',
                'f1_score',
                'precision',
                'accuracy',
                'R2',
            ]
        ]:
            self.fac = 1
        elif True in [
            m in self.metric_to_track
            for m in ['hamming_loss', 'RMSE', 'MSE', 'MAE', 'RRMSE', 'loss']
        ]:
            self.fac = -1
        else:
            AttributeError(
                'Invalid metric name used for early stopping: '
                + str(self.metric_to_track)
            )


    def __call__(
        self,
        performance_results,
        model,
        epoch,
        optimizer_state_dict = None
    ):

        score = performance_results['val_' + self.metric_to_track] * self.fac
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_model = model
            self.best_performance_results = performance_results
            self.best_optimizer_state_dict = optimizer_state_dict
            # self.save_checkpoint(val_loss, model)

        elif (score <= self.best_score + self.delta) and self.use_early_stopping:
            self.counter += 1
            if self.verbose:
                print(
                    f'-----------------------------EarlyStopping counter: {self.counter} out of {self.patience}---------------------- best epoch currently {self.best_epoch}'
                )
            if self.counter >= self.patience:
                self.early_stop_flag = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_model = model
            self.best_performance_results = performance_results
            self.best_optimizer_state_dict = optimizer_state_dict
            self.counter = 0