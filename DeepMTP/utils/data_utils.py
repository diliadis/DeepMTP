import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from torch.utils.data import Dataset

def process_interaction_data(data, verbose=False):

    info = None

    if isinstance(data, pd.DataFrame) and len(data.columns) == 3:
        if verbose:
            print('Interaction file: triplet format detected')

        # check if the interaction file has the correct header format
        if set(data.columns) != set({'instance_id', 'target_id', 'value'}):
            raise Exception('Interaction file: The dataset has no header. Please supply an interaction file with a header of the following format <instance_id, target_id, score>')
        
        info = {'data': data, 'original_format': 'triplets'}

        # check the type of the instance_id values (int for normal ids, string for image directories)
        if pd.api.types.is_integer_dtype(data.dtypes['instance_id']):
            info['instance_id_type'] = 'int'
        elif pd.api.types.is_string_dtype(data.dtypes['instance_id']):
            info['instance_id_type'] = 'string'
        else:
            raise TypeError('Instance_id type is not detected')

        # check the type of the target_id values (int for normal ids, string for image directories)
        if pd.api.types.is_integer_dtype(data.dtypes['target_id']):
            info['target_id_type'] = 'int'
        elif pd.api.types.is_string_dtype(data.dtypes['target_id']):
            info['target_id_type'] = 'string'
        else:
            raise TypeError('Target_id type is not detected')

    else:
        if verbose:
            print('Interaction file: 2d numpy array format detected')

        # converting the 2d numpy array format to the more flexible triplet format
        triplets = [
            (i, j, data[i, j])
            for i in range(data.shape[0])
            for j in range(data.shape[1])
            if ((data[i, j] is not None) and (not np.isnan(data[i, j])))
        ]
        data = pd.DataFrame(triplets, columns=["instance_id", "target_id", "value"])
        info = {'data': data,
                'original_format': 'numpy',
                'instance_id_type': 'int',
                'target_id_type': 'int'
        }
    
    return info


def check_interaction_files_format(data, verbose=False):

    distinct_formats = set(
        [
            data[mode]['original_format']
            for mode in ['y_train', 'y_test', 'y_val'] if data[mode] is not None
        ]
    )

    if distinct_formats:
        if verbose:
            print('Interaction file: checking format consistency... ', end='')

        if len(distinct_formats) == 1:
            print('Passed')
        else:
            raise Exception('Failed: Inconsistent file formats across the different interaction files')

def check_interaction_files_column_type_format(data, verbose=False):
    # check for instances
    distinct_instance_id_format = set(
        [
            data[mode]["instance_id_type"]
            for mode in ['y_train', 'y_test', 'y_val'] if data[mode] is not None
        ]
    )
    if distinct_instance_id_format:
        if verbose:
            print('Interaction file: checking instance id format consistency... ', end='')

        if len(distinct_instance_id_format) == 1:
            if verbose:
                print('Passed')
        else:
            raise Exception('Failed: Inconsistent instance id column type across the different interaction files')

    # check for targets
    distinct_target_id_format = set(
        [
            data[mode]["target_id_type"]
            for mode in ['y_train', 'y_test', 'y_val'] if data[mode] is not None
        ]
    )
    if distinct_target_id_format:
        if verbose:
            print('Interaction file: checking target id type consistency... ', end='')

        if len(distinct_target_id_format) == 1:
            if verbose:
                print('Passed')
        else:
            raise Exception('Failed: Inconsistent target id column type across the different interaction files')

def check_variable_type(samples_arr):
    variable_type = 'real-valued'
    if set(samples_arr['value']).difference(set([0, 1])) == set():
        variable_type = 'binary'
    return variable_type

def check_target_variable_type(data, verbose=False):

    distinct_target_variable_type = set(
        [
            check_variable_type(data[mode]['data'])
            for mode in ['y_train', 'y_test', 'y_val'] if data[mode] is not None
        ]
    )

    if distinct_target_variable_type:
        if verbose:
            print('Interaction file: checking target variable type consistency... ', end='')

        if len(distinct_target_variable_type) == 1:
            if verbose:
                print('Passed')
        else:
            raise Exception('Failed: Inconsistent target variable type across the different interaction files')

    # return the detected type of target variable
    return list(distinct_target_variable_type)[0]

def check_novel_instances(train_dict, test_dict, verbose):
    novel_instances_detected = None
    if test_dict is not None:
        if verbose:
            print('Interaction file: Checking for novel instances... ', end='')
        novel_instances_detected = False

        train_unique_instances = set(train_dict['data']['instance_id'])
        test_unique_instances = set(test_dict['data']['instance_id'])

        if train_dict['original_format'] == 'triplets':
            if train_unique_instances.intersection(test_unique_instances) == set():
                novel_instances_detected = True
        else:
            # this is a not so clever way to infer novel instances when your interaction data had a 2d numpy format...
            if len(train_unique_instances) != len(test_unique_instances):
                novel_instances_detected = True
        print('Done')
    return novel_instances_detected

def check_novel_targets(train_dict, test_dict, verbose):
    novel_targets_detected = None
    if test_dict is not None:
        if verbose:
            print('Interaction file: Checking for novel targets... ', end='')
        novel_targets_detected = False

        train_unique_targets = set(train_dict['data']['target_id'])
        test_unique_targets = set(test_dict['data']['target_id'])

        if train_dict['original_format'] == 'triplets':
            if train_unique_targets.intersection(test_unique_targets) == set():
                novel_targets_detected = True
        else:
            # this is a not so clever way to infer novel instances when your interaction data had a 2d numpy format...
            if len(train_unique_targets) != len(test_unique_targets):
                novel_targets_detected = True
        print('Done')
    return novel_targets_detected

def get_estimated_validation_setting(novel_instances, novel_targets, verbose):
    validation_setting_detected = None
    if novel_instances is not None and novel_targets is not None:
        if verbose:
            print('Calculating validation setting... ', end='')
        if novel_instances and not novel_targets:
            validation_setting_detected= "B"
        elif not novel_instances and novel_targets:
            validation_setting_detected = "C"
        elif novel_instances and novel_targets:
            validation_setting_detected = "D"
        elif not novel_instances and not novel_targets:
            validation_setting_detected = "A"
        print('Done', end='')
    
    if verbose:
        if validation_setting_detected is not None:
            print('-- Detected as setting :'+validation_setting_detected)
    return validation_setting_detected

def process_instance_features(train_instance_features, test_instance_features, val_instance_features, verbose=False):
    if verbose:
        print('Instance features file: processing features... ', end='')

    if train_instance_features is not None:
        train_instance_features = {'data': train_instance_features, 'num_features': train_instance_features.shape[1]}
    if test_instance_features is not None:
        test_instance_features = {'data': test_instance_features, 'num_features': test_instance_features.shape[1]}
    if val_instance_features is not None:
        val_instance_features = {'data': val_instance_features, 'num_features': val_instance_features.shape[1]}
    
    if verbose:
        print('Done')

    return train_instance_features, test_instance_features, val_instance_features

def process_target_features(train_target_features, test_target_features, val_target_features, verbose=False):
    if verbose:
        print('Target features file: processing features... ', end='')

    if train_target_features is not None:
        train_target_features = {'data': train_target_features, 'num_features': train_target_features.shape[1]}
    if test_target_features is not None:
        test_target_features = {'data': test_target_features, 'num_features': test_target_features.shape[1]}
    if val_target_features is not None:
        val_target_features = {'data': val_target_features, 'num_features': val_target_features.shape[1]}
    
    if verbose:
        print('Done')

    return train_target_features, test_target_features, val_target_features

def cross_input_consistency_check_instances(data, validation_setting, verbose):

    num_interaction_data_sources = sum([interaction_data is not None for interaction_data in [data['y_train'], data['y_test'], data['y_val']]])
    num_instance_features_sources = sum([interaction_data is not None for interaction_data in [data['X_train_instance'], data['X_test_instance'], data['X_val_instance']]])
    valid_modes = [mode for mode in ['train', 'test', 'val'] if data['y_'+mode] is not None]

    if num_instance_features_sources != 0:
        # if the interaction files have a numpy format, the only case currently allowed is the following: There are as many interaction files as there are feature files
        if 'numpy' in set([interaction_data['original_format'] for interaction_data in [data['y_train'], data['y_test'], data['y_val']] if interaction_data is not None]):
            if validation_setting in ['B', 'D']:
                if num_interaction_data_sources != num_instance_features_sources:
                    raise Exception('Different number of (numpy) interaction files and instance feature files is not currently supported')

                else:
                    if verbose:
                        print("Cross input consistency for (numpy) interaction data and instance features checks out")

                    if validation_setting == 'B':
                        for mode in valid_modes:
                            unique_entities_in_interactions_file = set(data['y_'+mode]['data']['instance_id'])
                            unique_entities_in_features_file = set(range(data['X_'+mode+'_instance']["data"].shape[0]))
                            if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                                if verbose:
                                    print('-- Same instance ids in the interaction and features files for the '+mode+' set')
                            else:
                                raise Exception('Different instance ids in the interaction and features files for the '+mode+' set.')

            elif validation_setting in ['C', 'A']:
                if num_instance_features_sources != 1:
                    raise Exception('Setting '+validation_setting+' needs only one instance feature file')
                else:
                    if verbose:
                        print('Cross input consistency for (numpy) interaction data and instance features checks out')

                    unique_entities_in_interactions_file = set()
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = unique_entities_in_interactions_file.union(set(data['y_'+mode]['data']['instance_id']))
                    unique_entities_in_features_file = set(range(data['X_train_instance']["data"].shape[0]))
                    if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                        if verbose:
                            print('-- Same instance ids in the interaction and features file')
                    else:
                        raise Exception('Different instance ids in the interaction and features files')

        else:
            '''
            if the interaction files have a triplet format, only two cases are currently allowed
                1.There are as many interaction files as there are feature files
                2.There are multiple interaction files and a single feature file
            '''

            if validation_setting in ['B', 'D']:
                if num_instance_features_sources > num_interaction_data_sources:
                    raise Exception('More instance feature files than (triplet) interaction files provided. This is currently not supported')
                
                elif num_instance_features_sources == num_interaction_data_sources:
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = set(data['y_'+mode]['data']['instance_id'])
                        unique_entities_in_features_file = set(range(data['X_'+mode+'_instance']["data"].shape[0]))
                        if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                            if verbose:
                                print('-- Same instance ids in the interaction and features files for the '+mode+' set')
                        else:
                            raise Exception('Different instance ids in the interaction and features files for the '+mode+' set')
                else:
                    if num_interaction_data_sources == 3 and num_instance_features_sources == 2:
                        raise Exception('When three (triplet) interaction files are provided, only one instance features file is permitted')
                    else:
                        unique_entities_in_interactions_file = set()
                        for mode in valid_modes:
                            unique_entities_in_interactions_file = unique_entities_in_interactions_file.union(set(data['y_'+mode]['data']['instance_id']))
                        unique_entities_in_features_file = set(range(data['X_train_instance']["data"].shape[0]))
                        if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                            if verbose:
                                print('-- Same instance ids in the interaction and features file')
                        else:
                            raise Exception('Different instance ids in the interaction and features files')

            elif validation_setting in ['C', 'A']:
                if num_instance_features_sources != 1:
                    raise Exception('When three (triplet) interaction files are provided with setting '+validation_setting+', only one instance features file is permitted')
                else:
                    if verbose:
                        print('-- Cross input consistency for (triplet) interaction data and instance features checks out')

                    unique_entities_in_interactions_file = set()
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = unique_entities_in_interactions_file.union(set(data['y_'+mode]['data']['instance_id']))
                    unique_entities_in_features_file = set(range(data['X_train_instance']["data"].shape[0]))
                    if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                        if verbose:
                            print('-- Same instance ids in the interaction and features file')
                    else:
                        raise Exception('Different instance ids in the interaction and features files')


def cross_input_consistency_check_targets(data, validation_setting, verbose):

    num_interaction_data_sources = sum([interaction_data is not None for interaction_data in [data['y_train'], data['y_test'], data['y_val']]])
    num_target_features_sources = sum([interaction_data is not None for interaction_data in [data['X_train_target'], data['X_test_target'], data['X_val_target']]])
    valid_modes = [mode for mode in ['train', 'test', 'val'] if data['y_'+mode] is not None]

    if num_target_features_sources != 0:

        # if the interaction files have a numpy format, the only case currently allowed is the following: There are as many interaction files as there are feature files
        if 'numpy' in set([interaction_data['original_format'] for interaction_data in [data['y_train'], data['y_test'], data['y_val']] if interaction_data is not None]):

            if validation_setting in ['B', 'D']:
                if num_interaction_data_sources != num_target_features_sources:
                    raise Exception('Different number of (numpy) interaction files and target feature files is not currently supported')

                else:
                    if verbose:
                        print("Cross input consistency for (numpy) interaction data and target features checks out")

                    if validation_setting == 'B':
                        for mode in valid_modes:
                            unique_entities_in_interactions_file = set(data['y_'+mode]['data']['target_id'])
                            unique_entities_in_features_file = set(range(data['X_'+mode+'_target']["data"].shape[0]))
                            if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                                if verbose:
                                    print('-- Same target ids in the interaction and features files for the '+mode+' set')
                            else:
                                raise Exception('Different target ids in the interaction and features files for the '+mode+' set.')

            elif validation_setting in ['C', 'A']:
                if num_target_features_sources != 1:
                    raise Exception('Setting '+validation_setting+' needs only one target feature file')
                else:
                    if verbose:
                        print('Cross input consistency for (numpy) interaction data and target features checks out')

                    unique_entities_in_interactions_file = set()
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = unique_entities_in_interactions_file.union(set(data['y_'+mode]['data']['target_id']))
                    unique_entities_in_features_file = set(range(data['X_train_target']["data"].shape[0]))
                    if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                        if verbose:
                            print('-- Same target ids in the interaction and features file')
                    else:
                        raise Exception('Different target ids in the interaction and features files')

        else:
            '''
            if the interaction files have a triplet format, only two cases are currently allowed
                1.There are as many interaction files as there are feature files
                2.There are multiple interaction files and a single feature file
            '''

            if validation_setting in ['B', 'D']:
                if num_target_features_sources > num_interaction_data_sources:
                    raise Exception('More target feature files than (triplet) interaction files provided. This is currently not supported')
                
                elif num_target_features_sources == num_interaction_data_sources:
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = set(data['y_'+mode]['data']['target_id'])
                        unique_entities_in_features_file = set(range(data['X_'+mode+'_target']["data"].shape[0]))
                        if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                            if verbose:
                                print('-- Same target ids in the interaction and features files for the '+mode+' set')
                        else:
                            raise Exception('Different target ids in the interaction and features files for the '+mode+' set')
                else:
                    if num_interaction_data_sources == 3 and num_target_features_sources == 2:
                        raise Exception('When three (triplet) interaction files are provided, only one target features file is permitted')
                    else:
                        unique_entities_in_interactions_file = set()
                        for mode in valid_modes:
                            unique_entities_in_interactions_file = unique_entities_in_interactions_file.union(set(data['y_'+mode]['data']['target_id']))
                        unique_entities_in_features_file = set(range(data['X_train_target']["data"].shape[0]))
                        if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                            if verbose:
                                print('-- Same target ids in the interaction and features file')
                        else:
                            raise Exception('Different target ids in the interaction and features files')

            elif validation_setting in ['C', 'A']:
                if num_target_features_sources != 1:
                    raise Exception('When three (triplet) interaction files are provided with setting '+validation_setting+', only one target features file is permitted')
                else:
                    if verbose:
                        print('Cross input consistency for (triplet) interaction data and target features checks out')

                    unique_entities_in_interactions_file = set()
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = unique_entities_in_interactions_file.union(set(data['y_'+mode]['data']['target_id']))
                    unique_entities_in_features_file = set(range(data['X_train_target']["data"].shape[0]))
                    if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                        if verbose:
                            print('-- Same target ids in the interaction and features file')
                    else:
                        raise Exception('Different target ids in the interaction and features files')

def split_data(data, validation_setting, split_method, ratio, seed, verbose):
    '''
    This function splits the dataset and offers two main functionalities:
    1) split based on the 4 different validation settings (A, B, C, D)
    2) if a test set already exists it separates a validation set, otherwise it first creates a test set and then a validation set
    '''

    data['y_train']['data']['old_instance_id'] = data['y_train']['data']['instance_id']
    data['y_train']['data']['old_target_id'] = data['y_train']['data']['target_id']
    if data['y_test'] is not None:
        data['y_test']['data']['old_instance_id'] = data['y_test']['data']['instance_id']
        data['y_test']['data']['old_target_id'] = data['y_test']['data']['target_id']
    if data['y_val'] is not None:
        data['y_val']['data']['old_instance_id'] = data['y_val']['data']['instance_id']
        data['y_val']['data']['old_target_id'] = data['y_val']['data']['target_id']

    if validation_setting == 'B':

        if data['y_test'] is None:
            if verbose:
                print('Splitting train to train-test according to validation setting '+validation_setting+'... ', end='')

            train_ids, test_ids = train_test_split(data['y_train']['data']['instance_id'].unique(), test_size=ratio['test'], random_state=seed)
            train_ids.sort()
            test_ids.sort()
            data['y_test'] = {'data': data['y_train']['data'][data['y_train']['data']['instance_id'].isin(test_ids)]}
            data['y_train']['data'] = data['y_train']['data'][data['y_train']['data']['instance_id'].isin(train_ids)]

            data['X_test_instance'] = {'data': data['X_train_instance']['data'][test_ids]}
            data['X_train_instance']['data'] = data['X_train_instance']['data'][train_ids]

            old_to_new_test_ids_map = {old_id: new_id for new_id, old_id in enumerate(test_ids)}
            old_to_new_train_ids_map = {old_id: new_id for new_id, old_id in enumerate(train_ids)}
            # data['y_train']['data'].replace({'instance_id': old_to_new_train_ids_map}, inplace=True)
            # data['y_test']['data'].replace({'instance_id': old_to_new_test_ids_map}, inplace=True)
            data['y_train']['data']['instance_id'] = data['y_train']['data']['instance_id'].map(old_to_new_train_ids_map)
            data['y_test']['data']['instance_id'] = data['y_test']['data']['instance_id'].map(old_to_new_test_ids_map)

            if verbose:
                print('Done')

        if data['y_val'] is None:
            if verbose:
                print('Splitting train to train-val according to validation setting '+validation_setting+'... ', end='')
            
            train_ids, val_ids = train_test_split(data['y_train']['data']['instance_id'].unique(), test_size=ratio['val'], random_state=seed)
            train_ids.sort()
            val_ids.sort()
            data['y_val'] = {'data': data['y_train']['data'][data['y_train']['data']['instance_id'].isin(val_ids)]}
            data['y_train']['data'] = data['y_train']['data'][data['y_train']['data']['instance_id'].isin(train_ids)]

            data['X_val_instance'] = {'data': data['X_train_instance']['data'][val_ids]}
            data['X_train_instance']['data'] = data['X_train_instance']['data'][train_ids]

            old_to_new_val_ids_map = {old_id: new_id for new_id, old_id in enumerate(val_ids)}
            old_to_new_train_ids_map = {old_id: new_id for new_id, old_id in enumerate(train_ids)}
            # data['y_train']['data'].replace({'instance_id': old_to_new_train_ids_map}, inplace=True)
            # data['y_val']['data'].replace({'instance_id': old_to_new_val_ids_map}, inplace=True)
            data['y_train']['data']['instance_id'] = data['y_train']['data']['instance_id'].map(old_to_new_train_ids_map)
            data['y_val']['data']['instance_id'] = data['y_val']['data']['instance_id'].map(old_to_new_val_ids_map)

            if verbose:
                print('Done')

    elif validation_setting == 'C':

        if data['y_test'] is None:
            if verbose:
                print('Splitting train to train-test according to validation setting '+validation_setting+'... ', end='')

            train_ids, test_ids = train_test_split(data['y_train']['data']['target_id'].unique(), test_size=ratio['test'], random_state=seed)
            train_ids.sort()
            test_ids.sort()
            data['y_test'] = {'data': data['y_train']['data'][data['y_train']['data']['target_id'].isin(test_ids)]}
            data['y_train']['data'] = data['y_train']['data'][data['y_train']['data']['target_id'].isin(train_ids)]

            data['X_test_target'] = {'data': data['X_train_target']['data'][test_ids]}
            data['X_train_target']['data'] = data['X_train_target']['data'][train_ids]

            old_to_new_test_ids_map = {old_id: new_id for new_id, old_id in enumerate(test_ids)}
            old_to_new_train_ids_map = {old_id: new_id for new_id, old_id in enumerate(train_ids)}
            # data['y_train']['data'].replace({'target_id': old_to_new_train_ids_map}, inplace=True)
            # data['y_test']['data'].replace({'target_id': old_to_new_test_ids_map}, inplace=True)
            data['y_train']['data']['target_id'] = data['y_train']['data']['target_id'].map(old_to_new_train_ids_map)
            data['y_test']['data']['target_id'] = data['y_test']['data']['target_id'].map(old_to_new_test_ids_map)

            if verbose:
                print('Done')

        if data['y_val'] is None:
            if verbose:
                print('Splitting train to train-val according to validation setting '+validation_setting+'... ', end='')

            train_ids, val_ids = train_test_split(data['y_train']['data']['target_id'].unique(), test_size=ratio['val'], random_state=seed)
            train_ids.sort()
            val_ids.sort()
            data['y_val'] = {'data': data['y_train']['data'][data['y_train']['data']['target_id'].isin(val_ids)]}
            data['y_train']['data'] = data['y_train']['data'][data['y_train']['data']['target_id'].isin(train_ids)]

            data['X_val_target'] = {'data': data['X_train_target']['data'][val_ids]}
            data['X_train_target']['data'] = data['X_train_target']['data'][train_ids]

            old_to_new_val_ids_map = {old_id: new_id for new_id, old_id in enumerate(val_ids)}
            old_to_new_train_ids_map = {old_id: new_id for new_id, old_id in enumerate(train_ids)}
            # data['y_train']['data'].replace({'target_id': old_to_new_train_ids_map}, inplace=True)
            # data['y_val']['data'].replace({'target_id': old_to_new_val_ids_map}, inplace=True)
            data['y_train']['data']['target_id'] = data['y_train']['data']['instance_id'].map(old_to_new_train_ids_map)
            data['y_val']['data']['target_id'] = data['y_val']['data']['instance_id'].map(old_to_new_val_ids_map)

            if verbose:
                print('Done')

    elif validation_setting == 'A':

        if data['y_test'] is None:
            if verbose:
                print('Splitting train to train-test according to validation setting '+validation_setting+'... ', end='')
            data['y_test'] = {'data': data['y_train']['data'].sample(frac=ratio['test'], replace=False, random_state=seed)}
            data['y_train']['data'] = data['y_train']['data'].drop(data['y_test']['data'].index)
            if verbose:
                print('Done')

        if data['y_val'] is None:
            if verbose:
                print('Splitting train to train-val according to validation setting '+validation_setting+'... ', end='')
            data['y_val'] = {'data': data['y_train']['data'].sample(frac=ratio['val'], replace=False, random_state=seed)}
            data['y_train']['data'] = data['y_train']['data'].drop(data['y_val']['data'].index)
            if verbose:
                print('Done')

    elif validation_setting == 'D':

        if data['y_test'] is None:
            if verbose:
                print('Splitting train to train-test according to validation setting '+validation_setting+'... ', end='')

            train_instance_ids, test_instance_ids = train_test_split(data['y_train']['data']['instance_id'].unique(), test_size=ratio['test'], random_state=seed)
            train_instance_ids.sort()
            test_instance_ids.sort()
            train_target_ids, test_target_ids = train_test_split(data['y_train']['data']['target_id'].unique(), test_size=ratio['test'], random_state=seed)
            train_target_ids.sort()
            test_target_ids.sort()

            data['y_test'] = {'data': data['y_train']['data'][ (data['y_train']['data']['instance_id'].isin(test_instance_ids)) & (data['y_train']['data']['target_id'].isin(test_target_ids)) ]}
            data['y_train']['data'] = data['y_train']['data'][ (data['y_train']['data']['instance_id'].isin(train_instance_ids)) & (data['y_train']['data']['target_id'].isin(train_target_ids)) ]

            data['X_test_instance'] = {'data': data['X_train_instance']['data'][test_instance_ids]}
            data['X_train_instance']['data'] = data['X_train_instance']['data'][train_instance_ids]

            data['X_test_target'] = {'data': data['X_train_target']['data'][test_target_ids]}
            data['X_train_target']['data'] = data['X_train_target']['data'][train_target_ids]

            old_to_new_test_instance_ids_map = {old_id: new_id for new_id, old_id in enumerate(test_instance_ids)}
            old_to_new_train_instance_ids_map = {old_id: new_id for new_id, old_id in enumerate(train_instance_ids)}
            data['y_train']['data']['instance_id'] = data['y_train']['data']['instance_id'].map(old_to_new_train_instance_ids_map)
            data['y_test']['data']['instance_id'] = data['y_test']['data']['instance_id'].map(old_to_new_test_instance_ids_map)

            old_to_new_test_target_ids_map = {old_id: new_id for new_id, old_id in enumerate(test_target_ids)}
            old_to_new_train_target_ids_map = {old_id: new_id for new_id, old_id in enumerate(train_target_ids)}
            data['y_train']['data']['target_id'] = data['y_train']['data']['target_id'].map(old_to_new_train_target_ids_map)
            data['y_test']['data']['target_id'] = data['y_test']['data']['target_id'].map(old_to_new_test_target_ids_map)

            if verbose:
                print('Done')

        if data['y_val'] is None:
            if verbose:
                print('Splitting train to train-val according to validation setting '+validation_setting+'... ', end='')

            train_instance_ids, val_instance_ids = train_test_split(data['y_train']['data']['instance_id'].unique(), test_size=ratio['val'], random_state=seed)
            train_instance_ids.sort()
            val_instance_ids.sort()
            train_target_ids, val_target_ids = train_test_split(data['y_train']['data']['target_id'].unique(), test_size=ratio['val'], random_state=seed)
            train_target_ids.sort()
            val_target_ids.sort()

            data['y_val'] = {'data': data['y_train']['data'][ (data['y_train']['data']['instance_id'].isin(val_instance_ids)) & (data['y_train']['data']['target_id'].isin(val_target_ids)) ]}
            data['y_train']['data'] = data['y_train']['data'][ (data['y_train']['data']['instance_id'].isin(train_instance_ids)) & (data['y_train']['data']['target_id'].isin(train_target_ids)) ]

            data['X_val_instance'] = {'data': data['X_train_instance']['data'][val_instance_ids]}
            data['X_train_instance']['data'] = data['X_train_instance']['data'][train_instance_ids]

            data['X_val_target'] = {'data': data['X_train_target']['data'][val_target_ids]}
            data['X_train_target']['data'] = data['X_train_target']['data'][train_target_ids]

            old_to_new_val_instance_ids_map = {old_id: new_id for new_id, old_id in enumerate(val_instance_ids)}
            old_to_new_train_instance_ids_map = {old_id: new_id for new_id, old_id in enumerate(train_instance_ids)}
            data['y_train']['data']['instance_id'] = data['y_train']['data']['instance_id'].map(old_to_new_train_instance_ids_map)
            data['y_val']['data']['instance_id'] = data['y_val']['data']['instance_id'].map(old_to_new_val_instance_ids_map)

            old_to_new_val_target_ids_map = {old_id: new_id for new_id, old_id in enumerate(val_target_ids)}
            old_to_new_train_target_ids_map = {old_id: new_id for new_id, old_id in enumerate(train_target_ids)}
            data['y_train']['data']['target_id'] = data['y_train']['data']['target_id'].map(old_to_new_train_target_ids_map)
            data['y_val']['data']['target_id'] = data['y_val']['data']['target_id'].map(old_to_new_val_target_ids_map)

            if verbose:
                print('Done')


def data_process(data, validation_setting=None, split_method='random', ratio={'train': 0.7, 'test': 0.2, 'val': 0.1}, seed=42, verbose=False):
    data = data.copy()
    train_flag, test_flag, val_flag = False, False, False
    setting_A_flag, setting_B_flag, setting_C_flag, setting_D_flag = None, None, None, None

    if validation_setting is not None :
        validation_setting = validation_setting.upper()

    # check if at least one train dataset exists (Can be only the y_train)
    if data['y_train'] is None:
        raise AttributeError('Passing y_train is the minimum requirement for creating a dataset')
    
    
    '''
    check if the user provides only the training data without a validation setting. 
    In this case we cannot automatically infer the validation setting.

    Also instance or target side information without the corresponding score matrices doesn't make sense
    '''
    if data['y_test'] is None:
        if validation_setting is None:
            raise AttributeError('The validation setting must be specified manually. To automatically infer it you must pass the test set as well!!')

        if data['X_test_instance'] is not None or data['X_test_target'] is not None:
             warnings.warn("Warning: You provided side information for the test set without the interaction matrix. This info won't be used")
    else:
        if data['X_train_instance'] is not None and data['X_test_instance'] is None:
            raise AttributeError('Train instance features are available but not test instance features')
        if data['X_train_target'] is not None and data['X_test_target'] is None:
            raise AttributeError('Train target features are available but not test target features')

    if data['y_val'] is None:
        if data['X_val_instance'] is not None or data['X_val_target'] is not None:
             warnings.warn("Warning: You provided side information for the validation set without the interaction matrix. This info won't be used")
    else:
        if data['X_train_instance'] is not None and data['X_val_instance'] is None:
            raise AttributeError('Train instance features are available but not validation instance features')
        if data['X_train_target'] is not None and data['X_val_target'] is None:
            raise AttributeError('Train target features are available but not validation target features')

    # check if the specified validation setting makes sense given the supplied datasets
    if validation_setting is not None:
        if validation_setting == 'B' and data['X_train_instance'] is None:
            raise Exception('Specified validation setting B without supplying instance features')
        elif validation_setting == 'C' and data['X_train_target'] is None:
            raise Exception('Specified validation setting C without supplying instance features')
        elif validation_setting == 'D' and data['X_train_instance'] is None: 
            raise Exception('Specified validation setting D without supplying instance features')
        elif validation_setting == 'D' and data['X_train_target'] is None:
            raise Exception('Specified validation setting D without supplying target features')

    # process the interaction data. The data dictionary will be augmented with additional inferred info
    data['y_train'] = process_interaction_data(data['y_train'], verbose=verbose)
    if data['y_test'] is not None:
        data['y_test'] = process_interaction_data(data['y_test'], verbose=verbose)
    if data['y_val'] is not None:
        data['y_val'] = process_interaction_data(data['y_val'], verbose=verbose)

    
    # check for format consistency across the interaction files
    check_interaction_files_format(data, verbose=verbose)
    # check for format consistency for instance and target ids across the interaction files
    check_interaction_files_column_type_format(data, verbose=verbose)

    print('')
    # get the type of the target variable.
    target_variable_type = check_target_variable_type(data, verbose=verbose)
    if verbose:
        print('Automatically detected type of target variable type: '+target_variable_type+'\n')


    # check for novel instances
    novel_instances = check_novel_instances(data['y_train'], data['y_test'], verbose=verbose)
    if verbose: 
        if novel_instances is None:
            print('-- Test set was not provided, could not detect if novel instances exist or not ')
        elif novel_instances:
            print('-- Novel instances detected in the test set')
        else:
            print('-- no Novel instances detected in the test set')

    # check for novel targets
    novel_targets = check_novel_targets(data['y_train'], data['y_test'], verbose=verbose)
    if verbose:
        if novel_targets is None:
            print('-- Test set was not provided, could not detect if novel targets exist or not ')
        elif novel_targets:
            print('-- Novel targets detected in the test set')
        else:
            print('-- no Novel targets detected in the test set')

    # use the information about the existence of novel instances and targets to infer the validation setting
    validation_setting_detected = get_estimated_validation_setting(novel_instances, novel_targets, verbose)
    print('')
    data['X_train_instance'], data['X_test_instance'], data['X_val_instance'] = process_instance_features(data['X_train_instance'], data['X_test_instance'], data['X_val_instance'], verbose=verbose)
    data['X_train_target'], data['X_test_target'], data['X_val_target'] = process_target_features(data['X_train_target'], data['X_test_target'], data['X_val_target'], verbose=verbose)
    
    print('')
    if validation_setting is None:
        if validation_setting_detected is not None:
            validation_setting = validation_setting_detected
        else:
            raise Exception('Validation setting was both not provided and not detected')
    else:
        if validation_setting_detected == 'D':
            print('Detected validation setting D as a possibility but will use the one defined by the user: setting '+validation_setting)
        # elif validation_setting_detected != validation_setting:
        #     raise Exception('Mismatch between the auto-detected validation setting and the one defined by the user --> User: '+validation_setting+' != Auto-detected: '+validation_setting_detected) 
    data['info'] = {'detected_validation_setting': validation_setting}
    data['info']['detected_problem_mode'] = 'classification' if target_variable_type == 'binary' else 'regression'
    if data['X_train_instance'] is not None:
        data['info']['instance_branch_input_dim'] = data['X_train_instance']['num_features']
    else:
        data['info']['instance_branch_input_dim'] = len(data['y_train']['data']['instance_id'].unique())
    if data['X_train_target'] is not None:
        data['info']['target_branch_input_dim'] = data['X_train_target']['num_features']
    else:
        data['info']['target_branch_input_dim'] = len(data['y_train']['data']['target_id'].unique())

    cross_input_consistency_check_instances(data, validation_setting, verbose)
    cross_input_consistency_check_targets(data, validation_setting, verbose)

    print('')
    split_data(data, validation_setting=validation_setting, split_method=split_method, ratio=ratio, seed=seed, verbose=verbose)

    train_data = {k.replace('_train', ''): v for k,v in data.items() if 'train' in k}
    val_data = {k.replace('_val', ''): v for k,v in data.items() if 'val' in k}
    test_data = {k.replace('_test', ''): v for k,v in data.items() if 'test' in k}

    return train_data, val_data, test_data, data['info']

class BaseDataset(Dataset):

    def __init__(self, config, data, instance_features, target_features, dyadic_features=None):
        self.config = config
        self.instance_branch_input_dim = config['instance_branch_input_dim']
        self.target_branch_input_dim = config['target_branch_input_dim']

        self.triplet_data = data['data']
        self.instance_features = None
        self.target_features = None
        if instance_features is not None:
            self.instance_features = instance_features['data']
        else:
            self.instance_features = np.zeros((self.instance_branch_input_dim, self.instance_branch_input_dim), int)
            np.fill_diagonal(self.instance_features, 1)

        if target_features is not None:
            self.target_features = target_features['data']
        else:
            self.target_features = np.zeros((self.target_branch_input_dim, self.target_branch_input_dim), int)
            np.fill_diagonal(self.target_features, 1)


    def __len__(self):
        return len(self.triplet_data)

    def __getitem__(self, idx): 
        instance_id = int(self.triplet_data.iloc[idx]['instance_id'])
        target_id = int(self.triplet_data.iloc[idx]['target_id'])
        value = self.triplet_data.iloc[idx]['value']
        instance_features_vec = None
        target_features_vec = None

        instance_features_vec = self.instance_features[instance_id]
        target_features_vec = self.target_features[target_id]

        return{'instance_id': instance_id, 'target_id': target_id, 'instance_features': instance_features_vec, 'target_features': target_features_vec, 'score': value}

'''
class BaseDataset(Dataset):

    def __init__(self, config, data, instance_features, target_features, dyadic_features=None):
        self.config = config
        self.instance_branch_input_dim = config['instance_branch_input_dim']
        self.target_branch_input_dim = config['target_branch_input_dim']

        self.triplet_data = data['data']
        self.instance_features = None
        self.target_features = None
        if instance_features is not None:
            self.instance_features = instance_features['data']
        if target_features is not None:
            self.target_features = target_features['data']

        self.dyadic_features = dyadic_features
        if dyadic_features is not None:
            self.dyadic_branch_input_dim = config['dyadic_branch_input_dim']

    def __len__(self):
        return len(self.triplet_data)

    def __getitem__(self, idx): 
        instance_id = int(self.triplet_data.iloc[idx]['instance_id'])
        target_id = int(self.triplet_data.iloc[idx]['target_id'])
        value = self.triplet_data.iloc[idx]['value']
        instance_features_vec = None
        target_features_vec = None
        dyadic_features_vec = None

        if self.instance_features is not None:
            instance_features_vec = self.instance_features[instance_id]
        else:
            instance_features_vec = np.zeros(self.instance_branch_input_dim)
            instance_features_vec[instance_id] = 1

        if self.target_features is not None:
            target_features_vec = self.target_features[target_id]
        else:
            target_features_vec = np.zeros(self.target_branch_input_dim) 
            target_features_vec[target_id] = 1

        if self.dyadic_features is not None:
            dyadic_features_vec = self.dyadic_features[instance_id, target_id]


        return{'instance_id': instance_id, 'target_id': target_id, 'instance_features': instance_features_vec, 'target_features': target_features_vec, 'score': value}
'''