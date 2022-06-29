import numpy as np

def check_mlc_results(train, val, test):
    '''Checks if the number of instance ids between the interaction data and instance features are the same.

    Args:
        train (dict): The training dataset
        val (dict): The validation dataset
        test (dict): the test dataset 
    '''
    print('Checking if MLC splitting results are valid... ', end='')
    # checks if the instance ids in the interaction and instance feature datasets are exactly the same 
    assert set(train['X_instance']['data']['id'].unique()).difference(set(train['y']['data']['instance_id'].unique())) == set()
    assert set(val['X_instance']['data']['id'].unique()).difference(set(val['y']['data']['instance_id'].unique())) == set()
    assert set(test['X_instance']['data']['id'].unique()).difference(set(test['y']['data']['instance_id'].unique())) == set()

    # check if train, val and test instance feature datasets don't share any instance_ids
    assert set(train['X_instance']['data']['id'].unique()).union(set(val['X_instance']['data']['id'].unique()))
    assert set(val['X_instance']['data']['id'].unique()).union(set(test['X_instance']['data']['id'].unique()))
    assert set(train['X_instance']['data']['id'].unique()).union(set(test['X_instance']['data']['id'].unique()))

    print('Done')

def check_mtr_results(train, val, test):
    '''Checks if the number of instance ids between the interaction data and instance features are the same.

    Args:
        train (dict): The training dataset
        val (dict): The validation dataset
        test (dict): the test dataset 
    '''
    print('Checking if MTR splitting results are valid... ', end='')
    # checks if the instance ids in the interaction and instance feature datasets are exactly the same 
    assert set(train['X_instance']['data']['id'].unique()).difference(set(train['y']['data']['instance_id'].unique())) == set()
    assert set(val['X_instance']['data']['id'].unique()).difference(set(val['y']['data']['instance_id'].unique())) == set()
    assert set(test['X_instance']['data']['id'].unique()).difference(set(test['y']['data']['instance_id'].unique())) == set()

    # check if train, val and test instance feature datasets don't share any instance_ids
    assert set(train['X_instance']['data']['id'].unique()).union(set(val['X_instance']['data']['id'].unique()))
    assert set(val['X_instance']['data']['id'].unique()).union(set(test['X_instance']['data']['id'].unique()))
    assert set(train['X_instance']['data']['id'].unique()).union(set(test['X_instance']['data']['id'].unique()))
    print('Done')

def check_dp_results(train, val, test):
    '''Checks if the number of instance ids between the interaction data and instance features are the same.
       Also checks if the number of target ids between the interaction data and target features are the same.
    Args:
        train (dict): The training dataset
        val (dict): The validation dataset
        test (dict): the test dataset 
    '''
    print('Checking if DP splitting results are valid... ', end='')
    # checks if the instance ids in the interaction and instance feature datasets are exactly the same 
    assert set(train['X_instance']['data']['id'].unique()).symmetric_difference(set(train['y']['data']['instance_id'].unique())) == set()
    assert set(val['X_instance']['data']['id'].unique()).symmetric_difference(set(val['y']['data']['instance_id'].unique())) == set()
    assert set(test['X_instance']['data']['id'].unique()).symmetric_difference(set(test['y']['data']['instance_id'].unique())) == set()
    # checks if the target ids in the interaction and target feature datasets are exactly the same 
    assert set(train['X_target']['data']['id'].unique()).symmetric_difference(set(train['y']['data']['target_id'].unique())) == set()
    assert set(val['X_target']['data']['id'].unique()).symmetric_difference(set(val['y']['data']['target_id'].unique())) == set()
    assert set(test['X_target']['data']['id'].unique()).symmetric_difference(set(test['y']['data']['target_id'].unique())) == set()
    print('Done')

def check_mc_results(train, val, test):
    '''Checks if the pairs of instance, target ids in the train, validation and test sets are unique

    Args:
        train (dict): The training dataset
        val (dict): The validation dataset
        test (dict): the test dataset 
    '''
    print('Checking if MC splitting results are valid... ', end='')
    # check if train, val and test don't share any <instance_id, target_id> pairs
    assert set(list(map(tuple, np.array(train['y']['data'][['instance_id', 'target_id']])))).intersection(set(list(map(tuple, np.array(val['y']['data'][['instance_id', 'target_id']]))))) == set()
    assert set(list(map(tuple, np.array(train['y']['data'][['instance_id', 'target_id']])))).intersection(set(list(map(tuple, np.array(test['y']['data'][['instance_id', 'target_id']]))))) == set()
    assert set(list(map(tuple, np.array(val['y']['data'][['instance_id', 'target_id']])))).intersection(set(list(map(tuple, np.array(test['y']['data'][['instance_id', 'target_id']]))))) == set()

def check_mtl_results(train, val, test):
    '''Checks if the number of instance ids between the interaction data and instance features are the same.

    Args:
        train (dict): The training dataset
        val (dict): The validation dataset
        test (dict): the test dataset 
    '''
    print('Checking if MTR splitting results are valid... ', end='')
    # checks if the instance ids in the interaction and instance feature datasets are exactly the same 
    assert set(train['X_instance']['data']['id'].unique()).difference(set(train['y']['data']['instance_id'].unique())) == set()
    assert set(val['X_instance']['data']['id'].unique()).difference(set(val['y']['data']['instance_id'].unique())) == set()
    assert set(test['X_instance']['data']['id'].unique()).difference(set(test['y']['data']['instance_id'].unique())) == set()

    # check if train, val and test instance feature datasets don't share any instance_ids
    assert set(train['X_instance']['data']['id'].unique()).union(set(val['X_instance']['data']['id'].unique()))
    assert set(val['X_instance']['data']['id'].unique()).union(set(test['X_instance']['data']['id'].unique()))
    assert set(train['X_instance']['data']['id'].unique()).union(set(test['X_instance']['data']['id'].unique()))
    print('Done')