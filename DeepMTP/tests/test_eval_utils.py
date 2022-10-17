from DeepMTP.utils.eval_utils import get_performance_results
import numpy as np
from sklearn.metrics import hamming_loss, f1_score, recall_score, precision_score

def test_get_performance_results():
    
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
        'classification',
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
    
    assert results['train_accuracy_macro'] == results['train_accuracy_micro']
    assert results['train_accuracy_macro'] == macro_accuracy
    assert results['train_hamming_loss_macro'] == results['train_hamming_loss_micro']
    assert results['train_hamming_loss_macro'] == macro_hamming_loss
    assert results['train_f1_score_macro'] == macro_f1_score
    assert results['train_f1_score_micro'] == micro_f1_score
    assert results['train_recall_macro'] == macro_recall
    assert results['train_recall_micro'] == micro_recall
    assert results['train_precision_macro'] == macro_precision
    assert results['train_precision_micro'] == micro_precision
