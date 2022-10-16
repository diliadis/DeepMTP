from DeepMTP.dataset import load_process_MLC, load_process_MTR, load_process_DP, load_process_MC, load_process_MTL

def test_load_process_MLC():
    variant = 'undivided'
    features_type = 'numpy'
    data = load_process_MLC(path='./data', dataset_name='bibtex', variant=variant, features_type=features_type, print_mode='basic')
    
    if variant == 'undivided':
        assert isinstance(data['val']['y'] , type(None))
        assert isinstance(data['val']['X_instance'] , type(None))
        assert isinstance(data['val']['X_target'] , type(None))
        assert isinstance(data['test']['y'] , type(None))
        assert isinstance(data['test']['X_instance'] , type(None))
        assert isinstance(data['test']['X_target'] , type(None))
        assert isinstance(data['train']['X_target'] , type(None))
    
    else:
        assert isinstance(data['train']['X_target'] , type(None))
        assert isinstance(data['val']['y'] , type(None))
        assert isinstance(data['val']['X_instance'] , type(None))
        assert isinstance(data['val']['X_target'] , type(None))
        assert isinstance(data['test']['X_target'] , type(None))

    if features_type == 'numpy':
        assert data['train']['y'].shape[0] == data['train']['X_instance'].shape[0]
        if variant != 'undivided':
            assert data['test']['y'].shape[0] == data['train']['X_instance'].shape[0]
            
    else:
        assert data['train']['y'].shape[0] == len(data['train']['X_instance'])
        if variant != 'undivided':
            assert data['test']['y'].shape[0] == len(data['train']['X_instance'])
            
def test_load_process_MTR():
    variant = 'undivided'
    features_type = 'numpy'
    data = load_process_MTR(path='./data', dataset_name='enb', features_type='numpy', print_mode='basic')
    
    if variant == 'undivided':
        assert isinstance(data['val']['y'] , type(None))
        assert isinstance(data['val']['X_instance'] , type(None))
        assert isinstance(data['val']['X_target'] , type(None))
        assert isinstance(data['test']['y'] , type(None))
        assert isinstance(data['test']['X_instance'] , type(None))
        assert isinstance(data['test']['X_target'] , type(None))
        assert isinstance(data['train']['X_target'] , type(None))
    
    else:
        assert isinstance(data['train']['X_target'] , type(None))
        assert isinstance(data['val']['y'] , type(None))
        assert isinstance(data['val']['X_instance'] , type(None))
        assert isinstance(data['val']['X_target'] , type(None))
        assert isinstance(data['test']['X_target'] , type(None))

    if features_type == 'numpy':
        assert data['train']['y'].shape[0] == data['train']['X_instance'].shape[0]
        if variant != 'undivided':
            assert data['test']['y'].shape[0] == data['train']['X_instance'].shape[0]
            
    else:
        assert data['train']['y'].shape[0] == len(data['train']['X_instance'])
        if variant != 'undivided':
            assert data['test']['y'].shape[0] == len(data['train']['X_instance'])
            
def test_load_process_DP():
    variant = 'undivided'
    validation_setting = 'B'
    data = load_process_DP(path='./data', dataset_name='ern', variant=variant, random_state=42, split_ratio={'train': 0.7, 'val': 0.1, 'test': 0.2}, split_instance_features=False, split_target_features=False, validation_setting='B', print_mode='basic')
    
    if validation_setting == 'B':
        assert data['train']['y'].shape[0] == data['train']['X_instance'].shape[0]
        assert data['val']['y'].shape[0] == data['val']['X_instance'].shape[0]
        assert data['test']['y'].shape[0] == data['test']['X_instance'].shape[0]
        assert data['train']['y'].shape[1] == data['val']['y'].shape[1]
        assert data['train']['y'].shape[1] == data['test']['y'].shape[1]
        assert data['train']['y'].shape[1] == data['train']['X_target'].shape[0]
        
    elif validation_setting == 'C':
        assert data['train']['y'].shape[1] == data['train']['X_target'].shape[0]
        assert data['val']['y'].shape[1] == data['val']['X_target'].shape[0]
        assert data['test']['y'].shape[1] == data['test']['X_target'].shape[0]
        assert data['train']['y'].shape[0] == data['val']['y'].shape[0]
        assert data['train']['y'].shape[0] == data['test']['y'].shape[0]
        assert data['train']['y'].shape[0] == data['train']['X_instance'].shape[1]

    elif validation_setting == 'D':
        assert data['train']['y'].shape[0] == data['train']['X_instance'].shape[0]
        assert data['val']['y'].shape[0] == data['val']['X_instance'].shape[0]
        assert data['test']['y'].shape[0] == data['test']['X_instance'].shape[0]
        assert data['train']['y'].shape[1] == data['train']['X_target'].shape[0]
        assert data['val']['y'].shape[1] == data['val']['X_target'].shape[0]
        assert data['test']['y'].shape[1] == data['test']['X_target'].shape[0]


def test_load_process_MC():
    data = load_process_MC(path='./data', dataset_name='ml-100k', print_mode='basic')
    
    assert isinstance(data['train']['X_instance'] , type(None))
    assert isinstance(data['train']['X_target'] , type(None))
    assert isinstance(data['val']['y'] , type(None))
    assert isinstance(data['val']['X_instance'] , type(None))
    assert isinstance(data['val']['X_target'] , type(None))
    assert isinstance(data['test']['X_instance'] , type(None))
    assert isinstance(data['test']['X_target'] , type(None))
    assert isinstance(data['test']['y'] , type(None))

def test_load_process_MTL():
    data = load_process_MTL(path='./data', dataset_name='dog', print_mode='basic') 
       
    assert isinstance(data['train']['X_target'] , type(None))
    assert isinstance(data['val']['y'] , type(None))
    assert isinstance(data['val']['X_instance'] , type(None))
    assert isinstance(data['val']['X_target'] , type(None))
    assert isinstance(data['test']['X_target'] , type(None))