import sys
sys.path.insert(0, '../../../..')
from DeepMTP.utils.utils import BaseExperimentInfo, get_optimization_direction

class RandomSearch:
    '''Implements the basic Random search HPO method. Nothing fancy, just a for loop over randomly generated configurations.
    '''
    def __init__(
        self,
        base_worker,
        configspace,
        budget=1,
        max_num_epochs=100,
        direction='min',
        verbose=False,
    ):
        self.base_worker = base_worker
        self.configspace = configspace
        self.verbose = verbose
        self.direction = direction
        self.budget = budget
        self.max_num_epochs = max_num_epochs
        self.experiment_history = {}

    def get_run_summary(self):
        return self.experiment_history

    def run_optimizer(self):

        self.configs_to_evaluate = [BaseExperimentInfo(config=self.configspace.sample_configuration(), budget=self.max_num_epochs) for c in range(self.budget)]

        for experiment in self.configs_to_evaluate:
            if self.verbose:
                print('---- Evaluating configuration... ')

            temp_result_dict = self.base_worker.compute(
                self.max_num_epochs, experiment.config
            )
                    
            experiment.score = temp_result_dict['loss']
            experiment.info = temp_result_dict['info']

            if self.verbose:
                print(
                    '---- Finished evaluating configuration with score: '
                    + str(experiment.score)
                )

        self.configs_to_evaluate = sorted(
            self.configs_to_evaluate,
            key=lambda x: x.score,
            reverse=False if self.direction == 'min' else True,
        )

        best_overall_config = self.configs_to_evaluate[0]
        best_overall_config.info['config']['experiment_name'] = 'best_model'

        if self.verbose:
            print('Best overall configuration: ')
            print(best_overall_config)
        return best_overall_config