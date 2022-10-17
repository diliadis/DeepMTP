from DeepMTP.utils.eval_utils import get_performance_results

def test_get_performance_results():
    
    y_true = np.random.rand(1000,10)
    y_true[y_true>0.5] = 1
    y_true[y_true<=0.5] = 0

    y_pred = np.random.rand(1000,10)
    y_pred[y_pred>0.5] = 1
    y_pred[y_pred<=0.5] = 0
    
    # calculating macro accuracy
    performance_per_column = []
    for j in range(y_true.shape[1]):
        result = y_true[:,j] == y_pred[:,j]
        performance_per_column.append(list(result.astype(int)).count(1)/y_true.shape[0])
        
    macro_accuracy = np.mean(performance_per_column)
    
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
        ['accuracy'],
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
    
    