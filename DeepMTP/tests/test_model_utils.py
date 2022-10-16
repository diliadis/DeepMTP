from DeepMTP.utils.model_utils import EarlyStopping

def test_early_stopping():
    
    # test 1
    early = EarlyStopping(True, patience=2, delta=0.0, metric_to_track='loss', verbose=True)
    for i, d in enumerate([0.6, 0.5, 0.4, 0.2, 0.1]):
        if not early.early_stop_flag:
            val_results = {'val_loss': d}
            early(
                val_results,
                None,
                i,
            )
        else:
            break
            
    assert early.get_best_epoch() == 4
    assert early.get_best_score() == -0.1
    
    # test 2
    early = EarlyStopping(True, patience=2, delta=0.0, metric_to_track='loss', verbose=True)
    for i, d in enumerate([0.6, 0.7, 0.8, 0.2, 0.1]):
        if not early.early_stop_flag:
            val_results = {'val_loss': d}
            early(
                val_results,
                None,
                i,
            )
        else:
            break
            
    assert early.get_best_epoch() == 0
    assert early.get_best_score() == -0.6
    
    # test 3
    early = EarlyStopping(True, patience=2, delta=0.0, metric_to_track='loss', verbose=True)
    for i, d in enumerate([0.6, 0.7, 0.5, 0.4, 0.9, 0.1]):
        if not early.early_stop_flag:
            val_results = {'val_loss': d}
            early(
                val_results,
                None,
                i,
            )
        else:
            break
            
    assert early.get_best_epoch() == 5
    assert early.get_best_score() == -0.1
    
    # test 4
    early = EarlyStopping(True, patience=3, delta=0.1, metric_to_track='loss', verbose=True)
    for i, d in enumerate([0.9, 0.8, 0.5, 0.4]):
        if not early.early_stop_flag:
            val_results = {'val_loss': d}
            early(
                val_results,
                None,
                i,
            )
        else:
            break
            
    assert early.get_best_epoch() == 3
    assert early.get_best_score() == -0.4
    
    # test 5
    early = EarlyStopping(True, patience=3, delta=0.11, metric_to_track='loss', verbose=True)
    for i, d in enumerate([0.9, 0.8, 0.6, 0.5, 0.5, 0.4]):
        if not early.early_stop_flag:
            val_results = {'val_loss': d}
            early(
                val_results,
                None,
                i,
            )
        else:
            break
            
    assert early.get_best_epoch() == 5
    assert early.get_best_score() == -0.4
    
    # test 6
    early = EarlyStopping(True, patience=2, delta=0.11, metric_to_track='loss', verbose=True)
    for i, d in enumerate([0.9, 0.8, 0.6, 0.5, 0.5, 0.4]):
        if not early.early_stop_flag:
            val_results = {'val_loss': d}
            early(
                val_results,
                None,
                i,
            )
        else:
            break
            
    assert early.get_best_epoch() == 4
    assert early.get_best_score() == -0.5