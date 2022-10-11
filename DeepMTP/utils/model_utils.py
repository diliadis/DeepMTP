import numpy as np

class EarlyStopping:
    '''Early stops the training if validation loss doesn't improve after a given patience.'''

    def __init__(self, use_early_stopping, patience=7, delta=0.0, metric_to_track='loss', verbose=False):
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
        
        self.delta_best_score = None
        self.delta_best_epoch = None
        self.delta_best_model = None
        self.delta_best_optimizer_state_dict = None
        self.delta_best_performance_results = None

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
                
            if not (score <= self.best_score):
                self.delta_best_score = score
                self.delta_best_epoch = epoch
                self.delta_best_model = model
                self.delta_best_performance_results = performance_results
                self.delta_best_optimizer_state_dict = optimizer_state_dict
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_model = model
            self.best_performance_results = performance_results
            self.best_optimizer_state_dict = optimizer_state_dict
            self.counter = 0
            
            self.delta_best_score = None
            self.delta_best_epoch = None
            self.delta_best_model = None
            self.delta_best_performance_results = None
            self.delta_best_optimizer_state_dict = None
            
            
    def get_best_score(self):
        if self.delta_best_score is None:
            return self.best_score
        else:
            return self.delta_best_score
        
    def get_best_epoch(self):
        if self.delta_best_epoch is None:
            return self.best_epoch
        else:
            return self.delta_best_epoch
        
    def get_best_model(self):
        if self.delta_best_model is None:
            return self.best_model
        else:
            return self.delta_best_model
        
    def get_best_performance_results(self):
        if self.delta_best_performance_results is None:
            return self.best_performance_results
        else:
            return self.delta_best_performance_results
        
    def get_best_optimizer_state_dict(self):
        if self.delta_best_optimizer_state_dict is None:
            return self.best_optimizer_state_dict
        else:
            return self.delta_best_optimizer_state_dict