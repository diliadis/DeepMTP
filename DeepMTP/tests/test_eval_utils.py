from DeepMTP.utils.eval_utils import get_performance_results
import numpy as np
from sklearn.metrics import hamming_loss, f1_score, recall_score, precision_score, mean_squared_error, mean_absolute_error, r2_score
import pytest
import math

modes = ['classification', 'regression']

@pytest.mark.parametrize('mode', modes)
def test_get_performance_results(mode):
    
    if mode == 'classification':
        y_true = np.random.rand(1000,10)
        y_true[y_true>0.5] = 1
        y_true[y_true<=0.5] = 0

        y_pred = np.random.rand(1000,10)
        y_pred[y_pred>0.5] = 1
        y_pred[y_pred<=0.5] = 0
        
        # calculating macro accuracy
        accuracy_per_column = []
        hamming_loss_per_column = []
        f1_score_per_column = []
        recall_per_column = []
        precision_per_column = []
        for j in range(y_true.shape[1]):
            result = y_true[:,j] == y_pred[:,j]
            accuracy_per_column.append(list(result.astype(int)).count(1)/y_true.shape[0])
            hamming_loss_per_column.append(hamming_loss(y_true[:,j], y_pred[:,j]))
            f1_score_per_column.append(f1_score(y_true[:,j], y_pred[:,j]))
            recall_per_column.append(recall_score(y_true[:,j], y_pred[:,j]))
            precision_per_column.append(precision_score(y_true[:,j], y_pred[:,j]))

        macro_accuracy = np.mean(accuracy_per_column)
        macro_hamming_loss = np.mean(hamming_loss_per_column)
        macro_f1_score = np.mean(f1_score_per_column)
        macro_recall = np.mean(recall_per_column)
        macro_precision = np.mean(precision_per_column)
        
        micro_f1_score = f1_score(y_true.flatten(), y_pred.flatten())
        micro_recall = recall_score(y_true.flatten(), y_pred.flatten())
        micro_precision = precision_score(y_true.flatten(), y_pred.flatten())

        instances_arr, targets_arr, true_values_arr, pred_values_arr = [], [], [], []
        for i in range(y_true.shape[0]):
            for j in range(y_true.shape[1]):
                instances_arr.append(i)
                targets_arr.append(j)
                true_values_arr.append(y_true[i, j])
                pred_values_arr.append(y_pred[i, j])

        
        results = get_performance_results(
            'train',
            0,
            instances_arr,
            targets_arr,
            true_values_arr,
            pred_values_arr,
            'B',
            mode,
            ['accuracy', 'hamming_loss', 'f1_score', 'recall', 'precision'],
            ['macro', 'micro'],
            slices_arr=None,
            verbose=False,
            per_target_verbose=False,
            per_instance_verbose=False,
            top_k=None,
            return_detailed_macro=False,
            train_true_value=None,
            scaler_per_target=None,
        )
        
        assert math.isclose(results['train_accuracy_macro'], results['train_accuracy_micro'])
        assert math.isclose(results['train_accuracy_macro'], macro_accuracy)
        assert math.isclose(results['train_hamming_loss_macro'], results['train_hamming_loss_micro'])
        assert math.isclose(results['train_hamming_loss_macro'], macro_hamming_loss)
        assert math.isclose(results['train_f1_score_macro'], macro_f1_score)
        assert math.isclose(results['train_f1_score_micro'], micro_f1_score)
        assert math.isclose(results['train_recall_macro'], macro_recall)
        assert math.isclose(results['train_recall_micro'], micro_recall)
        assert math.isclose(results['train_precision_macro'], macro_precision)
        assert math.isclose(results['train_precision_micro'], micro_precision)
        
    else:
        y_true = np.random.rand(1000,10)
        y_pred = np.random.rand(1000,10)

        # calculating macro metrics
        RMSE_per_column = []
        MSE_per_column = []
        MAE_score_per_column = []
        R2_per_column = []
        for j in range(y_true.shape[1]):
            RMSE_per_column.append(mean_squared_error(y_true[:,j], y_pred[:,j], squared=False))
            MSE_per_column.append(mean_squared_error(y_true[:,j], y_pred[:,j], squared=True))
            MAE_score_per_column.append(mean_absolute_error(y_true[:,j], y_pred[:,j]))
            R2_per_column.append(r2_score(y_true[:,j], y_pred[:,j]))
            
        macro_RMSE = np.mean(RMSE_per_column)
        macro_MSE = np.mean(MSE_per_column)
        macro_MAE = np.mean(MAE_score_per_column)
        macro_R2 = np.mean(R2_per_column)
        
        
        # calculating instance metrics
        RMSE_per_row = []
        MSE_per_row = []
        MAE_score_per_row = []
        R2_per_row = []
        for j in range(y_true.shape[0]):
            RMSE_per_row.append(mean_squared_error(y_true[j, :], y_pred[j, :], squared=False))
            MSE_per_row.append(mean_squared_error(y_true[j, :], y_pred[j, :], squared=True))
            MAE_score_per_row.append(mean_absolute_error(y_true[j, :], y_pred[j, :]))
            R2_per_row.append(r2_score(y_true[j, :], y_pred[j, :]))
            
        instance_RMSE = np.mean(RMSE_per_row)
        instance_MSE = np.mean(MSE_per_row)
        instance_MAE = np.mean(MAE_score_per_row)
        instance_R2 = np.mean(R2_per_row)
        
        # calculating metrics
        micro_RMSE = mean_squared_error(y_true.flatten(), y_pred.flatten(), squared=False)
        micro_R2 = r2_score(y_true.flatten(), y_pred.flatten())


        instances_arr, targets_arr, true_values_arr, pred_values_arr = [], [], [], []
        for i in range(y_true.shape[0]):
            for j in range(y_true.shape[1]):
                instances_arr.append(i)
                targets_arr.append(j)
                true_values_arr.append(y_true[i, j])
                pred_values_arr.append(y_pred[i, j])

        results = get_performance_results(
            'train',
            0,
            instances_arr,
            targets_arr,
            true_values_arr,
            pred_values_arr,
            'B',
            mode,
            ['RMSE', 'MSE', 'MAE', 'R2'],
            ['macro', 'micro', 'instance'],
            slices_arr=None,
            verbose=False,
            per_target_verbose=False,
            per_instance_verbose=False,
            top_k=None,
            return_detailed_macro=False,
            train_true_value=None,
            scaler_per_target=None,
        )
        
        assert math.isclose(results['train_MSE_macro'], results['train_MSE_micro'])
        assert math.isclose(results['train_MSE_macro'], macro_MSE)
        assert math.isclose(results['train_MSE_instance'], instance_MSE)
        
        assert math.isclose(results['train_MAE_macro'], results['train_MAE_micro'])
        assert math.isclose(results['train_MAE_macro'], macro_MAE)
        assert math.isclose(results['train_MAE_instance'], instance_MAE)

        assert math.isclose(results['train_R2_macro'], macro_R2)
        assert math.isclose(results['train_R2_micro'], micro_R2)
        assert math.isclose(results['train_R2_instance'], instance_R2)
        
        assert math.isclose(results['train_RMSE_macro'], macro_RMSE)
        assert math.isclose(results['train_RMSE_micro'], micro_RMSE)
        assert math.isclose(results['train_RMSE_instance'], instance_RMSE)
        