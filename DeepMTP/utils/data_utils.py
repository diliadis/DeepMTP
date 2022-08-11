import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
from PIL import Image
from torch.utils.data import Dataset
import copy

def normalize(row, scaler):
    ''' Just normalizes a row or features

    Args:
        row (numpy.array): an array of features 
        scaler (sklearn.scaler): the scaler that will be used to scale the features

    Returns:
        numpy.array: a scaled array of feature
    '''    
    return scaler.transform([row])[0]

def process_interaction_data(interaction_data, verbose=False, print_mode='basic'):
    '''A function that processes the interaction data. It is called separately for the train, val and test interaction data.
    There are two main types of interaction data formats that are supported:
    -> numpy format: This is a 2d numpy array that represents the interaction (also called score) matrix usually found in problems settings with fully observed matrices (multi-label classification, multivariate regression)
    -> triplet format: The most flexible format as it can be used to represent every possible problem setting

    Args:
        interaction_data (_type_): a numpy array or a dataframe with the interaction data
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    Returns:
        dict: a dictionary with the interaction data and additional information that could be detected (format, type of instance and target ids, etc.)
    '''    
    info = None
    if isinstance(interaction_data, pd.DataFrame):
        if len(interaction_data.columns) == 3:
            if verbose: print(('info: ' if print_mode=='dev' else '')+'Interaction file: triplet format detected')

            # check if the interaction file has the correct header format
            if set(interaction_data.columns) != set({'instance_id', 'target_id', 'value'}):
                raise Exception('Interaction file: The dataset has no header. Please supply an interaction file with a header of the following format <instance_id, target_id, score>')
            info = {'data': interaction_data, 'original_format': 'triplets'}
            if pd.api.types.is_string_dtype(interaction_data.dtypes['value']):
                raise AttributeError("The 'value' column in the dataframe with the interaction triplets should contain integers or floats, not strings.")

            # check the type of the instance_id values (int for normal ids, string for image directories)
            if pd.api.types.is_integer_dtype(interaction_data.dtypes['instance_id']):
                info['instance_id_type'] = 'int'
            elif pd.api.types.is_string_dtype(interaction_data.dtypes['instance_id']):
                info['instance_id_type'] = 'string'
            else:
                raise TypeError('Instance_id type is not detected')

            # check the type of the target_id values (int for normal ids, string for image directories)
            if pd.api.types.is_integer_dtype(interaction_data.dtypes['target_id']):
                info['target_id_type'] = 'int'
            elif pd.api.types.is_string_dtype(interaction_data.dtypes['target_id']):
                info['target_id_type'] = 'string'
            else:
                raise TypeError('Target_id type is not detected')
        else:
            raise Exception('Interaction file with invalid header detected. Please use the following header <instance_id, target_id, value>')
    elif isinstance(interaction_data, np.ndarray):
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Interaction file: 2d numpy array format detected')

        # converting the 2d numpy array format to the more flexible triplet format
        triplets = [
            (i, j, interaction_data[i, j])
            for i in range(interaction_data.shape[0])
            for j in range(interaction_data.shape[1])
            if ((interaction_data[i, j] is not None) and (not np.isnan(interaction_data[i, j])))
        ]
        interaction_data = pd.DataFrame(triplets, columns=['instance_id', 'target_id', 'value'])
        info = {'data': interaction_data,
                'original_format': 'numpy',
                'instance_id_type': 'int',
                'target_id_type': 'int'
        }
    else:
        raise Exception('Could not recognize the format of the interaction data. A 2d numpy array or a dataframe(<instance_id, target_id, value>) is currently supported')
    info['missing_values'] = False if len(info['data']) == (info['data']['instance_id'].nunique() * info['data']['target_id'].nunique()) else True
    return info


def check_interaction_files_format(data, verbose=False, print_mode='basic'):
    '''Checks the format of the interaction data. If any inconsistencies are detected (like different formats between train and test interaction data), an exception is raised

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    '''    
    distinct_formats = set([data[mode]['y']['original_format'] for mode in ['train', 'test', 'val'] if data[mode]['y'] is not None])
    if distinct_formats:
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Interaction file: checking format consistency... ', end='')
        if len(distinct_formats) == 1:
            print(('info: ' if print_mode=='dev' else '')+'Passed')
        else:
            raise Exception(('error: ' if print_mode=='dev' else '')+'Inconsistent file formats across the different interaction files')

def check_interaction_files_column_type_format(data, verbose=False, print_mode='basic'):
    '''Checks the type of the instance and target ids in the interaction data. If any inconsistencies are detected (like different id types between train and test interaction data), an exception is raised

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    '''    
    # check for instance id type concistency across the interaction data sources
    distinct_instance_id_format = set([data[mode]['y']['instance_id_type'] for mode in ['train', 'test', 'val'] if data[mode]['y'] is not None])
    if distinct_instance_id_format:
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Interaction file: checking instance id format consistency... ', end='')
        if len(distinct_instance_id_format) == 1:
            if verbose: print(('info: ' if print_mode=='dev' else '')+'Passed')
        else:
            raise Exception(('error: ' if print_mode=='dev' else '')+'Inconsistent instance id column type across the different interaction files')

    # check for target id type concistency across the interaction data sources
    distinct_target_id_format = set([data[mode]['y']['target_id_type'] for mode in ['train', 'test', 'val'] if data[mode]['y'] is not None])
    if distinct_target_id_format:
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Interaction file: checking target id type consistency... ', end='')
        if len(distinct_target_id_format) == 1:
            if verbose: print(('info: ' if print_mode=='dev' else '')+'Passed')
        else:
            raise Exception(('error: ' if print_mode=='dev' else '')+'Inconsistent target id column type across the different interaction files')

def check_variable_type(samples_arr):
    '''Checks the type of the target variable. This implementation is currently very simple. It has to be improved!!

    Args:
        samples_arr (numpy.array): A numpy array with the target variables 

    Returns:
        str: The type of the target variable. Possible values are 'real-valued' and 'binary'
    '''    
    variable_type = 'real-valued'
    if set(samples_arr['value']).difference(set([0, 1])) == set():
        variable_type = 'binary'
    return variable_type

def check_target_variable_type(data, verbose=False, print_mode='basic'):
    '''Checks the type of the target variable in the interaction data. If any inconsistencies are detected (like different id types between train and test interaction data), an exception is raised

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    '''   
    distinct_target_variable_type = set([check_variable_type(data[mode]['y']['data']) for mode in ['train', 'test', 'val'] if data[mode]['y'] is not None])
    if distinct_target_variable_type:
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Interaction file: checking target variable type consistency... ', end='')
        if len(distinct_target_variable_type) == 1:
            if verbose: print(('info: ' if print_mode=='dev' else '')+'Passed')
        else:
            raise Exception(('error: ' if print_mode=='dev' else '')+'Inconsistent target variable type across the different interaction files')
    # return the detected type of target variable
    return list(distinct_target_variable_type)[0]

def check_novel_instances(train, test, verbose=False, print_mode='basic'):
    '''Checks if the test set contains novel instance ids that are not present in the training set

    Args:
        train (dict): The actual train interaction data.
        test (dict): The actual train interaction data.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    '''   
    novel_instances_detected = None
    if test is not None:
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Interaction file: Checking for novel instances... ', end='')
        novel_instances_detected = False
        train_unique_instances = set(train['data']['instance_id'])
        test_unique_instances = set(test['data']['instance_id'])
        if train['original_format'] == 'triplets':
            if train_unique_instances.intersection(test_unique_instances) == set():
                novel_instances_detected = True
        else:
            # this is a not so clever way to infer novel instances when your interaction data had a 2d numpy format...
            if len(train_unique_instances) != len(test_unique_instances):
                novel_instances_detected = True
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Done')
    return novel_instances_detected

def check_novel_targets(train, test, verbose=False, print_mode='basic'):
    '''Checks if the test set contains novel target ids that are not present in the training set

    Args:
        train (dict): The actual train interaction data.
        test (dict): The actual train interaction data.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    '''  
    novel_targets_detected = None
    if test is not None:
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Interaction file: Checking for novel targets... ', end='')
        novel_targets_detected = False
        train_unique_targets = set(train['data']['target_id'])
        test_unique_targets = set(test['data']['target_id'])
        if train['original_format'] == 'triplets':
            if train_unique_targets.intersection(test_unique_targets) == set():
                novel_targets_detected = True
        else:
            # this is a not so clever way to infer novel instances when your interaction data had a 2d numpy format...
            if len(train_unique_targets) != len(test_unique_targets):
                novel_targets_detected = True
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Done')
    return novel_targets_detected

def get_estimated_validation_setting(novel_instances_flag, novel_targets_flag, verbose, print_mode='basic'):
    '''Uses the combination of infrmation about novel instances and novel targets to determine the validation setting that is possible.

    Args:
        novel_instances_flag (bool): A boolean that indicates whether or not the test set contains novel instances.
        novel_targets_flag (bool): A boolean that indicates whether or not the test set contains novel targets.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    Returns:
        str: The validation setting. Possible values are A, B, C, D
    '''
    validation_setting_detected = None
    if novel_instances_flag is not None and novel_targets_flag is not None:
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Estimating validation setting... ', end='')
        if novel_instances_flag and not novel_targets_flag:
            validation_setting_detected= 'B'
        elif not novel_instances_flag and novel_targets_flag:
            validation_setting_detected = 'C'
        elif novel_instances_flag and novel_targets_flag:
            validation_setting_detected = 'D'
        elif not novel_instances_flag and not novel_targets_flag:
            validation_setting_detected = 'A'
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Done', end='')
    if verbose and validation_setting_detected is not None: print(('info: ' if print_mode=='dev' else '')+'-- Detected as setting :'+validation_setting_detected)
    return validation_setting_detected

def process_instance_features(instance_features, verbose=False, print_mode='basic'):
    '''A function that processes the instance features data. It is called separately for the train, val and test interaction data.
    There are two main types of interaction data formats that are supported:
    * numpy format: This is a 2d numpy array that represents the instance features. The first dimension represents the samples and the second dimension the features.
    * triplet format: This is a dataframe with two columns. 
        * The first column stores the instance id (that will be mapped to the instance ids in the interaction data).
        * The second column stores the features that can have two different names:
            * 'features': stores the actuall features array
            * 'dir': stores the directory that the features are stored. (In the current implementation this is designed for image datasets but it can be generalized in future versions)

    Args:
        instance_features (_type_): The instance features
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    Returns:
        dict: A dictionary with the instance features and other related information that could be detected
    '''    
    if instance_features is not None:
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Instance features file: processing features... ', end='')
        if isinstance(instance_features, np.ndarray):
            temp_instance_features_df = pd.DataFrame(np.arange(len(instance_features)), columns=['id'])
            temp_instance_features_df['features'] = [r for r in instance_features]
            instance_features = temp_instance_features_df
            instance_features = {'data': instance_features, 'num_features': len(instance_features.loc[0, 'features']), 'info': 'numpy'}
        elif isinstance(instance_features, pd.DataFrame):
            instance_features.set_index('id', inplace=True, drop=False)
            if set(instance_features.columns) == set({'id', 'features'}):
                instance_features = {'data': instance_features, 'num_features': len(instance_features.loc[0, 'features']), 'info': 'dataframe'}
            elif set(instance_features.columns) == set({'id', 'dir'}):
                instance_features = {'data': instance_features, 'num_features': None, 'info': 'images'}
            else:
                raise AttributeError('Column names: '+str(instance_features.columns)+' in the instance features dataframe are not recognized')
        else:
            raise AttributeError('Could not recognize the format of the Instance features data. A 2d numpy array or a dataframe(<id, features>) is currently supported')
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Done')
    return instance_features

def process_target_features(target_features, verbose=False, print_mode='basic'):
    '''A function that processes the target features data. It is called separately for the train, val and test interaction data.
    There are two main types of interaction data formats that are supported:
    * numpy format: This is a 2d numpy array that represents the target features. The first dimension represents the samples and the second dimension the features.
    * triplet format: This is a dataframe with two columns. 
        * The first column stores the target id (that will be mapped to the target ids in the interaction data).
        * The second column stores the features that can have two different names:
            * 'features': stores the actuall features array
            * 'dir': stores the directory that the features are stored. (In the current implementation this is designed for image datasets but it can be generalized in future versions)

    Args:
        target_features (_type_): The target features
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    Returns:
        dict: A dictionary with the target features and other related information that could be detected
    '''    
    if target_features is not None:
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Instance features file: processing features... ', end='')
        if isinstance(target_features, np.ndarray):
            temp_target_features_df = pd.DataFrame(np.arange(len(target_features)), columns=['id'])
            temp_target_features_df['features'] = [r for r in target_features]
            target_features = temp_target_features_df
            target_features = {'data': target_features, 'num_features': len(target_features.loc[0, 'features']), 'info': 'numpy'}
        elif isinstance(target_features, pd.DataFrame):
            target_features.set_index('id', inplace=True, drop=False)
            if set(target_features.columns) == set({'id', 'features'}):
                target_features = {'data': target_features, 'num_features': len(target_features.loc[0, 'features']), 'info': 'dataframe'}
            elif set(target_features.columns) == set({'id', 'dir'}):
                target_features = {'data': target_features, 'num_features': None, 'info': 'images'}
            else:
                raise AttributeError('Column names: '+str(target_features.columns)+' in the target features dataframe are not recognized')
        else:
            raise AttributeError('Could not recognize the format of the Target features data. A 2d numpy array or a dataframe(<id, features>) is currently supported')
        if verbose: print(('info: ' if print_mode=='dev' else '')+'Done')
    return target_features

def cross_input_consistency_check_instances(data, validation_setting, verbose, print_mode='basic'):
    '''Checks the consistency of instance ids in the interaction data and instance features. The requirements to pass this check change depending on the format of the interaction data and instance features

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        validation_setting (str): The validation setting of the current problem.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    '''
    num_interaction_data_sources = sum([data[mode]['y'] is not None for mode in ['train', 'test', 'val']])
    num_instance_features_sources = sum([data[mode]['X_instance'] is not None for mode in ['train', 'test', 'val']])
    valid_modes = [mode for mode in ['train', 'test', 'val'] if data[mode]['y'] is not None]

    if num_instance_features_sources != 0:
        # if the interaction files have a numpy format, the only case currently allowed is the following: There are as many interaction files as there are feature files
        if 'numpy' in set([data[mode]['y']['original_format'] for mode in ['train', 'test', 'val'] if data[mode]['y'] is not None]):
            if validation_setting in ['B', 'D']:
                if num_interaction_data_sources != num_instance_features_sources:
                    raise Exception(('error: ' if print_mode=='dev' else '')+'Different number of (numpy) interaction files and instance feature files is not currently supported')
                else:
                    if verbose: print(('info: ' if print_mode=='dev' else '')+'Cross input consistency for (numpy) interaction data and instance features checks out')
                    if validation_setting == 'B':
                        for mode in valid_modes:
                            unique_entities_in_interactions_file = set(data[mode]['y']['data']['instance_id'])
                            unique_entities_in_features_file = set(data[mode]['X_instance']['data']['id'].unique())
                            if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                                if verbose: print(('info: ' if print_mode=='dev' else '')+'-- Same instance ids in the interaction and features files for the '+mode+' set')
                            else:
                                raise Exception(('error: ' if print_mode=='dev' else '')+'Different instance ids in the interaction and features files for the '+mode+' set.')

            elif validation_setting in ['C', 'A']:
                if num_instance_features_sources != 1:
                    raise Exception(('error: ' if print_mode=='dev' else '')+'Setting '+validation_setting+' needs only one instance feature file')
                else:
                    if verbose: print(('info: ' if print_mode=='dev' else '')+'Cross input consistency for (numpy) interaction data and instance features checks out')
                    unique_entities_in_interactions_file = set()
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = unique_entities_in_interactions_file.union(set(data[mode]['y']['data']['instance_id']))
                        unique_entities_in_features_file = set(data[mode]['X_instance']['data']['id'].unique())
                    if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                        if verbose: print(('info: ' if print_mode=='dev' else '')+'-- Same instance ids in the interaction and features file')
                    else:
                        raise Exception(('error: ' if print_mode=='dev' else '')+'Different instance ids in the interaction and features files')

        else:
            '''
            if the interaction files have a triplet format, only two cases are currently allowed
                1.There are as many interaction files as there are feature files
                2.There are multiple interaction files and a single feature file
            '''

            if validation_setting in ['B', 'D', 'A']:
                if num_instance_features_sources > num_interaction_data_sources:
                    raise Exception(('error: ' if print_mode=='dev' else '')+'More instance feature files than (triplet) interaction files provided. This is currently not supported')
                elif num_instance_features_sources == num_interaction_data_sources:
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = set(data[mode]['y']['data']['instance_id'])
                        unique_entities_in_features_file = set(data[mode]['X_instance']['data']['id'].unique())
                        if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                            if verbose: print(('info: ' if print_mode=='dev' else '')+'-- Same instance ids in the interaction and features files for the '+mode+' set')
                        else:
                            raise Exception(('error: ' if print_mode=='dev' else '')+'Different instance ids in the interaction and features files for the '+mode+' set')
                else:
                    if num_interaction_data_sources == 3 and num_instance_features_sources == 2:
                        raise Exception(('error: ' if print_mode=='dev' else '')+'When three (triplet) interaction files are provided, only one instance features file is allowed')
                    else:
                        unique_entities_in_interactions_file = set()
                        for mode in valid_modes:
                            unique_entities_in_interactions_file = unique_entities_in_interactions_file.union(set(data[mode]['y']['data']['instance_id']))
                        unique_entities_in_features_file = set(data['train']['X_instance']['data']['id'].unique())
                        if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                            if verbose: print(('info: ' if print_mode=='dev' else '')+'-- Same instance ids in the interaction and features file')
                        else:
                            raise Exception(('error: ' if print_mode=='dev' else '')+'Different instance ids in the interaction and features files')
            elif validation_setting == 'C':
                if num_instance_features_sources != 1:
                    raise Exception(('error: ' if print_mode=='dev' else '')+'When '+str(len(valid_modes))+' (triplet) interaction files are provided with setting '+validation_setting+', only one instance features file is allowed')
                else:
                    if verbose: print(('info: ' if print_mode=='dev' else '')+'-- Cross input consistency for (triplet) interaction data and instance features checks out')
                    unique_entities_in_interactions_file = set()
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = unique_entities_in_interactions_file.union(set(data[mode]['y']['data']['instance_id']))
                    unique_entities_in_features_file = set(data['train']['X_instance']['data']['id'].unique())
                    if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                        if verbose: print(('info: ' if print_mode=='dev' else '')+'-- Same instance ids in the interaction and features file')
                    else:
                        raise Exception(('error: ' if print_mode=='dev' else '')+'Different instance ids in the interaction and features files')

def cross_input_consistency_check_targets(data, validation_setting, verbose, print_mode='basic'):
    '''Checks the consistency of target ids in the interaction data and target features. The requirements to pass this check change depending on the format of the interaction data and target features

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        validation_setting (str): The validation setting of the current problem.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    '''
    num_interaction_data_sources = sum([data[mode]['y'] is not None for mode in ['train', 'test', 'val']])
    num_target_features_sources = sum([data[mode]['X_target'] is not None for mode in ['train', 'test', 'val']])
    valid_modes = [mode for mode in ['train', 'test', 'val'] if data[mode]['y'] is not None]

    if num_target_features_sources != 0:
        # if the interaction files have a numpy format, the only case currently allowed is the following: There are as many interaction files as there are feature files
        if 'numpy' in set([data[mode]['y']['original_format'] for mode in ['train', 'test', 'val'] if data[mode]['y'] is not None]):
            if validation_setting in ['B', 'D']:
                if num_interaction_data_sources != num_target_features_sources:
                    raise Exception(('error: ' if print_mode=='dev' else '')+'Different number of (numpy) interaction files and target feature files is not currently supported')
                else:
                    if verbose: print(('info: ' if print_mode=='dev' else '')+'Cross input consistency for (numpy) interaction data and target features checks out')
                    if validation_setting == 'B':
                        for mode in valid_modes:
                            unique_entities_in_interactions_file = set(data[mode]['y']['data']['target_id'])
                            unique_entities_in_features_file = set(data[mode]['X_target']['data']['id'].unique())
                            if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                                if verbose: print(('info: ' if print_mode=='dev' else '')+'-- Same target ids in the interaction and features files for the '+mode+' set')
                            else:
                                raise Exception(('error: ' if print_mode=='dev' else '')+'Different target ids in the interaction and features files for the '+mode+' set.')
            elif validation_setting in ['C', 'A']:
                if num_target_features_sources != 1:
                    raise Exception(('error: ' if print_mode=='dev' else '')+'Setting '+validation_setting+' needs only one target feature file')
                else:
                    if verbose: print(('info: ' if print_mode=='dev' else '')+'Cross input consistency for (numpy) interaction data and target features checks out')
                    unique_entities_in_interactions_file = set()
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = unique_entities_in_interactions_file.union(set(data[mode]['y']['data']['target_id']))
                    unique_entities_in_features_file = set(data[mode]['X_target']['data']['id'].unique())
                    if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                        if verbose: print(('info: ' if print_mode=='dev' else '')+'-- Same target ids in the interaction and features file')
                    else:
                        raise Exception(('error: ' if print_mode=='dev' else '')+'Different target ids in the interaction and features files')
        else:
            '''
            if the interaction files have a triplet format, only two cases are currently allowed
                1.There are as many interaction files as there are feature files
                2.There are multiple interaction files and a single feature file
            '''

            if validation_setting in ['B', 'D', 'A']:
                if num_target_features_sources > num_interaction_data_sources:
                    raise Exception(('error: ' if print_mode=='dev' else '')+'More target feature files than (triplet) interaction files provided. This is currently not supported')
                elif num_target_features_sources == num_interaction_data_sources:
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = set(data[mode]['y']['data']['target_id'])
                        unique_entities_in_features_file = set(data[mode]['X_target']['data']['id'].unique())
                        if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                            if verbose: print(('info: ' if print_mode=='dev' else '')+'-- Same target ids in the interaction and features files for the '+mode+' set')
                        else:
                            raise Exception(('error: ' if print_mode=='dev' else '')+'Different target ids in the interaction and features files for the '+mode+' set')
                else:
                    if num_interaction_data_sources == 3 and num_target_features_sources == 2:
                        raise Exception(('error: ' if print_mode=='dev' else '')+'When three (triplet) interaction files are provided, only one target features file is allowed')
                    else:
                        unique_entities_in_interactions_file = set()
                        for mode in valid_modes:
                            unique_entities_in_interactions_file = unique_entities_in_interactions_file.union(set(data[mode]['y']['data']['target_id']))
                        unique_entities_in_features_file = set(data['train']['X_target']['data']['id'].unique())
                        if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                            if verbose: print(('info: ' if print_mode=='dev' else '')+'-- Same target ids in the interaction and features file')
                        else:
                            raise Exception(('error: ' if print_mode=='dev' else '')+'Different target ids in the interaction and features files')
            elif validation_setting == 'C':
                if num_target_features_sources != 1:
                    raise Exception(('error: ' if print_mode=='dev' else '')+'When '+str(len(valid_modes))+' (triplet) interaction files are provided with setting '+validation_setting+', only one target features file is allowed')
                else:
                    if verbose: print(('info: ' if print_mode=='dev' else '')+'-- Cross input consistency for (triplet) interaction data and target features checks out')
                    unique_entities_in_interactions_file = set()
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = unique_entities_in_interactions_file.union(set(data[mode]['y']['data']['target_id']))
                    unique_entities_in_features_file = set(data['train']['X_target']['data']['id'].unique())
                    if unique_entities_in_interactions_file.symmetric_difference(unique_entities_in_features_file) == set():
                        if verbose: print(('info: ' if print_mode=='dev' else '')+'-- Same target ids in the interaction and features file')
                    else:
                        raise Exception(('error: ' if print_mode=='dev' else '')+'Different target ids in the interaction and features files')

def split_data(data, validation_setting, split_method, ratio, seed, verbose, print_mode='basic'):
    '''Splits the dataset and offers two main functionalities:
        * split based on the 4 different validation settings (A, B, C, D)
        * if a test set already exists it separates a validation set, otherwise it first creates a test set and then a validation set.

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        validation_setting (str): The validation setting of the current problem.
        split_method (str): The splitting method used. The current implementation only supports the 'random split' using a specific seed but a future goal is to also offer a stratified option.
        ratio (dict): The train, val and test ratios used to split the data. 
        seed (int): The seed used to initiate the randomized split
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    '''    

    if validation_setting == 'B':
        if data['test']['y'] is None:
            if verbose: print(('info: ' if print_mode=='dev' else '')+'Splitting train to train-test according to validation setting '+validation_setting+'... ', end='')
            train_ids, test_ids = train_test_split(data['train']['y']['data']['instance_id'].unique(), test_size=ratio['test'], random_state=seed)
            train_ids.sort()
            test_ids.sort()
            data['test']['y'] = {'data': data['train']['y']['data'][data['train']['y']['data']['instance_id'].isin(test_ids)]}
            data['train']['y']['data'] = data['train']['y']['data'][data['train']['y']['data']['instance_id'].isin(train_ids)]
           
            data['test']['X_instance'] = {'data': data['train']['X_instance']['data'][data['train']['X_instance']['data']['id'].isin(test_ids)]}
            data['train']['X_instance']['data'] = data['train']['X_instance']['data'][data['train']['X_instance']['data']['id'].isin(train_ids)]

            if verbose:print(('info: ' if print_mode=='dev' else '')+'Done')

        if data['val']['y'] is None:
            if verbose: print(('info: ' if print_mode=='dev' else '')+'Splitting train to train-val according to validation setting '+validation_setting+'... ', end='')
            train_ids, val_ids = train_test_split(data['train']['y']['data']['instance_id'].unique(), test_size=ratio['val'], random_state=seed)
            train_ids.sort()
            val_ids.sort()
            data['val']['y'] = {'data': data['train']['y']['data'][data['train']['y']['data']['instance_id'].isin(val_ids)]}
            data['train']['y']['data'] = data['train']['y']['data'][data['train']['y']['data']['instance_id'].isin(train_ids)]

            data['val']['X_instance'] = {'data': data['train']['X_instance']['data'][data['train']['X_instance']['data']['id'].isin(val_ids)]}
            data['train']['X_instance']['data'] = data['train']['X_instance']['data'][data['train']['X_instance']['data']['id'].isin(train_ids)]
            if data['test']['X_instance'] is None:
                data['test']['X_instance']['data'] = data['train']['X_instance']['data'][data['train']['X_instance']['data']['id'].isin(data['test']['X_instance']['id'].unique())]
                data['train']['X_instance']['data'] = data['train']['X_instance']['data'][~data['train']['X_instance']['data']['id'].isin(data['test']['X_instance']['id'].unique())]

            if verbose: print(('info: ' if print_mode=='dev' else '')+'Done')

    elif validation_setting == 'C':
        if data['test']['y'] is None:
            if verbose: print(('info: ' if print_mode=='dev' else '')+'Splitting train to train-test according to validation setting '+validation_setting+'... ', end='')
            train_ids, test_ids = train_test_split(data['train']['y']['data']['target_id'].unique(), test_size=ratio['test'], random_state=seed)
            train_ids.sort()
            test_ids.sort()
            data['test']['y'] = {'data': data['train']['y']['data'][data['train']['y']['data']['target_id'].isin(test_ids)]}
            data['train']['y']['data'] = data['train']['y']['data'][data['train']['y']['data']['target_id'].isin(train_ids)]

            data['test']['X_target'] = {'data': data['train']['X_target']['data'][data['train']['X_target']['data']['id'].isin(test_ids)]}
            data['train']['X_target']['data'] = data['train']['X_target']['data'][data['train']['X_target']['data']['id'].isin(train_ids)]

            if verbose: print(('info: ' if print_mode=='dev' else '')+'Done')

        if data['val']['y'] is None:
            if verbose: print(('info: ' if print_mode=='dev' else '')+'Splitting train to train-val according to validation setting '+validation_setting+'... ', end='')
            train_ids, val_ids = train_test_split(data['train']['y']['data']['target_id'].unique(), test_size=ratio['val'], random_state=seed)
            train_ids.sort()
            val_ids.sort()
            data['val']['y'] = {'data': data['train']['y']['data'][data['train']['y']['data']['target_id'].isin(val_ids)]}
            data['train']['y']['data'] = data['train']['y']['data'][data['train']['y']['data']['target_id'].isin(train_ids)]

            data['val']['X_target'] = {'data': data['train']['X_target']['data'][data['train']['X_target']['data']['id'].isin(val_ids)]}
            data['train']['X_target']['data'] = data['train']['X_target']['data'][data['train']['X_target']['data']['id'].isin(train_ids)]
            if data['test']['X_target'] is None:
                data['test']['X_target']['data'] = data['train']['X_target']['data'][data['train']['X_target']['data']['id'].isin(data['test']['X_target']['id'].unique())]
                data['train']['X_target']['data'] = data['train']['X_target']['data'][~data['train']['X_target']['data']['id'].isin(data['test']['X_target']['id'].unique())]

            if verbose: print(('info: ' if print_mode=='dev' else '')+'Done')

    elif validation_setting == 'A':
        if data['test']['y'] is None:
            if verbose: print(('info: ' if print_mode=='dev' else '')+'Splitting train to train-test according to validation setting '+validation_setting+'... ', end='')
            data['test']['y'] = {'data': data['train']['y']['data'].sample(frac=ratio['test'], replace=False, random_state=seed)}
            # data['test']['y'].update({k:v for k,v in data['train']['y'].items() if k!=data})
            data['train']['y']['data'] = data['train']['y']['data'].drop(data['test']['y']['data'].index)
            if data['train']['X_instance'] is not None:
                data['test']['X_instance'] = data['train']['X_instance'].copy()
            if data['train']['X_target'] is not None:
                data['test']['X_target'] = data['train']['X_target'].copy()
            if verbose: print(('info: ' if print_mode=='dev' else '')+'Done')

        if data['val']['y'] is None:
            if verbose: print(('info: ' if print_mode=='dev' else '')+'Splitting train to train-val according to validation setting '+validation_setting+'... ', end='')
            data['val']['y'] = {'data': data['train']['y']['data'].sample(frac=ratio['val'], replace=False, random_state=seed)}
            # data['val']['y'].update({k:v for k,v in data['train']['y'].items() if k!=data})
            data['train']['y']['data'] = data['train']['y']['data'].drop(data['val']['y']['data'].index)
            if data['train']['X_instance'] is not None:
                if data['test']['X_instance'] is None:
                    data['test']['X_instance'] = data['train']['X_instance'].copy()
                if data['val']['X_instance'] is None:                    
                    data['val']['X_instance'] = data['train']['X_instance'].copy()
            if data['train']['X_target'] is not None:
                if data['test']['X_target'] is None:
                    data['test']['X_target'] = data['train']['X_target'].copy()
                if data['val']['X_target'] is None:
                    data['val']['X_target'] = data['train']['X_target'].copy()
            if verbose: print(('info: ' if print_mode=='dev' else '')+'Done')

    elif validation_setting == 'D':
        if data['test']['y'] is None:
            if verbose: print(('info: ' if print_mode=='dev' else '')+'Splitting train to train-test according to validation setting '+validation_setting+'... ', end='')
            train_instance_ids, test_instance_ids = train_test_split(data['train']['y']['data']['instance_id'].unique(), test_size=ratio['test'], random_state=seed)
            train_instance_ids.sort()
            test_instance_ids.sort()
            train_target_ids, test_target_ids = train_test_split(data['train']['y']['data']['target_id'].unique(), test_size=ratio['test'], random_state=seed)
            train_target_ids.sort()
            test_target_ids.sort()
            data['test']['y'] = {'data': data['train']['y']['data'][ (data['train']['y']['data']['instance_id'].isin(test_instance_ids)) & (data['train']['y']['data']['target_id'].isin(test_target_ids)) ]}
            # data['test']['y'].update({k:v for k,v in data['train']['y'].items() if k!=data})
            data['train']['y']['data'] = data['train']['y']['data'][ (data['train']['y']['data']['instance_id'].isin(train_instance_ids)) & (data['train']['y']['data']['target_id'].isin(train_target_ids)) ]

            data['test']['X_instance'] = {'data': data['train']['X_instance']['data'][data['train']['X_instance']['data']['id'].isin(test_instance_ids)]}
            data['train']['X_instance']['data'] = data['train']['X_instance']['data'][data['train']['X_instance']['data']['id'].isin(train_instance_ids)]

            data['test']['X_target'] = {'data': data['train']['X_target']['data'][data['train']['X_target']['data']['id'].isin(test_target_ids)]}
            data['train']['X_target']['data'] = data['train']['X_target']['data'][data['train']['X_target']['data']['id'].isin(train_target_ids)]

            if verbose: print(('info: ' if print_mode=='dev' else '')+'Done')

        if data['val']['y'] is None:
            if verbose: print(('info: ' if print_mode=='dev' else '')+'Splitting train to train-val according to validation setting '+validation_setting+'... ', end='')
            train_instance_ids, val_instance_ids = train_test_split(data['train']['y']['data']['instance_id'].unique(), test_size=ratio['val'], random_state=seed)
            train_instance_ids.sort()
            val_instance_ids.sort()
            train_target_ids, val_target_ids = train_test_split(data['train']['y']['data']['target_id'].unique(), test_size=ratio['val'], random_state=seed)
            train_target_ids.sort()
            val_target_ids.sort()
            data['val']['y'] = {'data': data['train']['y']['data'][ (data['train']['y']['data']['instance_id'].isin(val_instance_ids)) & (data['train']['y']['data']['target_id'].isin(val_target_ids)) ]}
            # data['val']['y'].update({k:v for k,v in data['train']['y'].items() if k!=data})
            data['train']['y']['data'] = data['train']['y']['data'][ (data['train']['y']['data']['instance_id'].isin(train_instance_ids)) & (data['train']['y']['data']['target_id'].isin(train_target_ids)) ]

            data['val']['X_instance'] = {'data': data['train']['X_instance']['data'][data['train']['X_instance']['data']['id'].isin(val_instance_ids)]}
            data['train']['X_instance']['data'] = data['train']['X_instance']['data'][data['train']['X_instance']['data']['id'].isin(train_instance_ids)]
            if data['test']['X_instance'] is None:
                data['test']['X_instance']['data'] = data['train']['X_instance']['data'][data['train']['X_instance']['data']['id'].isin(data['test']['X_instance']['id'].unique())]
                data['train']['X_instance']['data'] = data['train']['X_instance']['data'][~data['train']['X_instance']['data']['id'].isin(data['test']['X_instance']['id'].unique())]

            data['val']['X_target'] = {'data': data['train']['X_target']['data'][data['train']['X_target']['data']['id'].isin(val_target_ids)]}
            data['train']['X_target']['data'] = data['train']['X_target']['data'][data['train']['X_target']['data']['id'].isin(train_target_ids)]
            if data['test']['X_target'] is None:
                data['test']['X_target']['data'] = data['train']['X_target']['data'][data['train']['X_target']['data']['id'].isin(data['test']['X_target']['id'].unique())]
                data['train']['X_target']['data'] = data['train']['X_target']['data'][~data['train']['X_target']['data']['id'].isin(data['test']['X_target']['id'].unique())]

            if verbose: print(('info: ' if print_mode=='dev' else '')+'Done')


def data_process(data, validation_setting=None, split_method='random', ratio={'train': 0.7, 'test': 0.2, 'val': 0.1}, seed=42, verbose=False, print_mode='basic', scale_instance_features=None, scale_target_features=None):
    '''The main function that handles all the preprocessing steps and checks needed to prepare the dataset to be used by the model.

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        validation_setting (str, optional): The validation setting of the current problem. Defaults to None.
        split_method (str, optional): The splitting method used. The current implementation only supports the 'random split' using a specific seed but a future goal is to also offer a stratified option. Defaults to 'random'.
        ratio (dict, optional): The train, val and test ratios used to split the data. Defaults to {'train': 0.7, 'test': 0.2, 'val': 0.1}.
        seed (int, optional): The seed used to initiate the randomized split. Defaults to 42.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.
        scale_instance_features (str, optional): The scaler used for the instance features. Possible values are 'MinMax' for the MinMax scaler and 'Standard' for the standard scaler. Defaults to None.
        scale_target_features (str, optional): The scaler used for the target features. Possible values are 'MinMax' for the MinMax scaler and 'Standard' for the standard scaler. Defaults to None.

    Returns:
        dict, dict, dict, dict: Four different dictionaries containing:
            * Train processed data
            * Test processed data
            * Validation processed data
            * general information about the datasets
    '''
    data = copy.deepcopy(data)
    train_flag, test_flag, val_flag = False, False, False
    setting_A_flag, setting_B_flag, setting_C_flag, setting_D_flag = None, None, None, None

    if validation_setting is not None :
        validation_setting = validation_setting.upper()

    if 'train' not in data:
        data['train'] = {'y': None, 'X_instance': None, 'X_target': None}
    else:
        if 'y' not in data['train']:
            data['train']['y'] = None
        if 'X_instance' not in data['train']:
            data['train']['X_instance'] = None
        if 'X_target' not in data['train']:
            data['train']['X_target'] = None
    
    if 'test' not in data:
        data['test'] = {'y': None, 'X_instance': None, 'X_target': None}
    else:
        if 'y' not in data['test']:
            data['test']['y'] = None
        if 'X_instance' not in data['test']:
            data['test']['X_instance'] = None
        if 'X_target' not in data['test']:
            data['test']['X_target'] = None
    
    if 'val' not in data:
        data['val'] = {'y': None, 'X_instance': None, 'X_target': None}
    else:
        if 'y' not in data['val']:
            data['val']['y'] = None
        if 'X_instance' not in data['val']:
            data['val']['X_instance'] = None
        if 'X_target' not in data['val']:
            data['val']['X_target'] = None

    # check if at least one train dataset exists (Can be only the y_train)
    if data['train']['y'] is None:
        raise AttributeError(('error: ' if print_mode=='dev' else '')+'Passing y_train is the minimum requirement for creating a dataset')

    if data['test']['y'] is None and data['val']['y'] is not None:
        raise Exception(('error: ' if print_mode=='dev' else '')+'Supplying a validation set without a test set is not currently supported!!')
    
    '''
    check if the user provides only the training data without a validation setting. 
    In this case we cannot automatically infer the validation setting.

    Also instance or target side information without the corresponding score matrices doesn't make sense
    '''
    if data['test']['y'] is None:
        if validation_setting is None:
            raise AttributeError(('error: ' if print_mode=='dev' else '')+'The validation setting must be specified manually. To automatically infer it you must pass the test set as well!!')
    else:
        if data['train']['X_instance'] is not None and data['test']['X_instance'] is None:
            raise AttributeError(('error: ' if print_mode=='dev' else '')+'Instance features are available for the train set  but not the test set')
        if data['train']['X_target'] is not None and data['test']['X_target'] is None:
            raise AttributeError(('error: ' if print_mode=='dev' else '')+'Target features are available for the train set  but not the test set')

    if data['val']['y'] is None:
        if data['val']['X_instance'] is not None or data['val']['X_target'] is not None:
             warnings.warn(('warning: ' if print_mode=='dev' else '')+"Warning: You provided side information for the validation set without the interaction matrix. This info won't be used")
    else:
        if data['train']['X_instance'] is not None and data['val']['X_instance'] is None:
            raise AttributeError(('error: ' if print_mode=='dev' else '')+'Instance features are available for the train set  but not the validation set')
        if data['train']['X_target'] is not None and data['val']['X_target'] is None:
            raise AttributeError(('error: ' if print_mode=='dev' else '')+'Target features are available for the train set  but not the validation set')

    # check if the specified validation setting makes sense given the supplied datasets
    if validation_setting is not None:
        if validation_setting == 'B' and data['train']['X_instance'] is None:
            raise Exception(('error: ' if print_mode=='dev' else '')+'Specified validation setting B without supplying instance features')
        elif validation_setting == 'C' and data['train']['X_target'] is None:
            raise Exception(('error: ' if print_mode=='dev' else '')+'Specified validation setting C without supplying instance features')
        elif validation_setting == 'D' and data['train']['X_instance'] is None: 
            raise Exception(('error: ' if print_mode=='dev' else '')+'Specified validation setting D without supplying instance features')
        elif validation_setting == 'D' and data['train']['X_target'] is None:
            raise Exception(('error: ' if print_mode=='dev' else '')+'Specified validation setting D without supplying target features')

    # process the interaction data. The data dictionary will be augmented with additional inferred info
    data['train']['y'] = process_interaction_data(data['train']['y'], verbose=verbose, print_mode=print_mode)
    if data['test']['y'] is not None:
        data['test']['y'] = process_interaction_data(data['test']['y'], verbose=verbose, print_mode=print_mode)
    if data['val']['y'] is not None:
        data['val']['y'] = process_interaction_data(data['val']['y'], verbose=verbose, print_mode=print_mode)

    # check for format consistency across the interaction files
    check_interaction_files_format(data, verbose=verbose, print_mode=print_mode)
    # check for format consistency for instance and target ids across the interaction files
    check_interaction_files_column_type_format(data, verbose=verbose, print_mode=print_mode)

    print('')
    # get the type of the target variable.
    target_variable_type = check_target_variable_type(data, verbose=verbose, print_mode=print_mode)
    if verbose:
        print(('info: ' if print_mode=='dev' else '')+'Automatically detected type of target variable type: '+target_variable_type+'\n')


    # check for novel instances
    novel_instances = check_novel_instances(data['train']['y'], data['test']['y'], verbose=verbose, print_mode=print_mode)
    if verbose: 
        if novel_instances is None:
            print(('info: ' if print_mode=='dev' else '')+'-- Test set was not provided, could not detect if novel instances exist or not ')
        elif novel_instances:
            print(('info: ' if print_mode=='dev' else '')+'-- Novel instances detected in the test set')
        else:
            print(('info: ' if print_mode=='dev' else '')+'-- no Novel instances detected in the test set')

    # check for novel targets
    novel_targets = check_novel_targets(data['train']['y'], data['test']['y'], verbose=verbose, print_mode=print_mode)
    if verbose:
        if novel_targets is None:
            print(('info: ' if print_mode=='dev' else '')+'-- Test set was not provided, could not detect if novel targets exist or not ')
        elif novel_targets:
            print(('info: ' if print_mode=='dev' else '')+'-- Novel targets detected in the test set')
        else:
            print(('info: ' if print_mode=='dev' else '')+'-- no Novel targets detected in the test set')

    # use the information about the existence of novel instances and targets to infer the validation setting
    validation_setting_detected = get_estimated_validation_setting(novel_instances, novel_targets, verbose, print_mode=print_mode)
    print('')

    # process the instance features
    data['train']['X_instance'] = process_instance_features(data['train']['X_instance'], verbose=verbose, print_mode=print_mode)
    data['test']['X_instance'] = process_instance_features(data['test']['X_instance'], verbose=verbose, print_mode=print_mode)
    data['val']['X_instance'] = process_instance_features(data['val']['X_instance'], verbose=verbose, print_mode=print_mode)
    # process the target features
    data['train']['X_target'] = process_target_features(data['train']['X_target'], verbose=verbose, print_mode=print_mode)
    data['test']['X_target'] = process_target_features(data['test']['X_target'], verbose=verbose, print_mode=print_mode)
    data['val']['X_target'] = process_target_features(data['val']['X_target'], verbose=verbose, print_mode=print_mode)

    print('')
    if validation_setting is None:
        if validation_setting_detected is not None:
            validation_setting = validation_setting_detected
        else:
            raise Exception(('error: ' if print_mode=='dev' else '')+'Validation setting was both not provided and not detected')
    else:
        if validation_setting_detected == 'D':
            print(('info: ' if print_mode=='dev' else '')+'Detected validation setting D as a possibility but will use the one defined by the user: setting '+validation_setting)
        # elif validation_setting_detected != validation_setting:
        #     raise Exception('Mismatch between the auto-detected validation setting and the one defined by the user --> User: '+validation_setting+' != Auto-detected: '+validation_setting_detected) 
    data['info'] = {'detected_validation_setting': validation_setting}
    data['info']['detected_problem_mode'] = 'classification' if target_variable_type == 'binary' else 'regression'
    if data['train']['X_instance'] is not None:
        data['info']['instance_branch_input_dim'] = data['train']['X_instance']['num_features']
    else:
        data['info']['instance_branch_input_dim'] = len(data['train']['y']['data']['instance_id'].unique())
    if data['train']['X_target'] is not None:
        data['info']['target_branch_input_dim'] = data['train']['X_target']['num_features']
    else:
        data['info']['target_branch_input_dim'] = len(data['train']['y']['data']['target_id'].unique())
        
    cross_input_consistency_check_instances(data, validation_setting, verbose, print_mode=print_mode)
    cross_input_consistency_check_targets(data, validation_setting, verbose, print_mode=print_mode)

    print('')
    split_data(data, validation_setting=validation_setting, split_method=split_method, ratio=ratio, seed=seed, verbose=verbose, print_mode=print_mode)

    if scale_instance_features is not None:
        if scale_instance_features == 'MinMax':
            instance_scaler = MinMaxScaler()
        elif scale_instance_features == 'Standard':
            instance_scaler = StandardScaler()
        
        if data['train']['X_instance'] is not None:
            if 'features' in data['train']['X_instance']['data'].columns:
                if verbose: print(('info: ' if print_mode=='dev' else '')+'Scaling instance features... ', end='')
                # fit the instance features scaler
                instance_scaler.fit(np.stack(data['train']['X_instance']['data']['features'].to_numpy()))
                # transform the instance features for the train,val and test sets
                data['train']['X_instance']['data']['features'] = data['train']['X_instance']['data'].apply(lambda row: normalize(row['features'], instance_scaler), axis=1)
                data['val']['X_instance']['data']['features'] = data['val']['X_instance']['data'].apply(lambda row: normalize(row['features'], instance_scaler), axis=1)
                data['test']['X_instance']['data']['features'] = data['test']['X_instance']['data'].apply(lambda row: normalize(row['features'], instance_scaler), axis=1)
                if verbose: print(('info: ' if print_mode=='dev' else '')+'Done')
            else:
                if verbose: print(('warning: ' if print_mode=='dev' else '')+'-- Requested '+scale_instance_features+' scaling for the instance features while supplying images. This step will be skipped')

    if scale_target_features is not None:
        if scale_target_features == 'MinMax':
            target_scaler = MinMaxScaler()
        elif scale_target_features == 'Standard':
            target_scaler = StandardScaler()

        if data['train']['X_target'] is not None:
            if 'features' in data['train']['X_target']['data'].columns:
                if verbose: print(('info: ' if print_mode=='dev' else '')+'Scaling target features... ', end='')
                # fit the instance features scaler
                target_scaler.fit(np.stack(data['train']['X_target']['data']['features'].to_numpy()))
                # transform the instance features for the train,val and test sets
                data['train']['X_target']['data']['features'] = data['train']['X_target']['data'].apply(lambda row: normalize(row['features'], target_scaler), axis=1)
                data['val']['X_target']['data']['features'] = data['val']['X_target']['data'].apply(lambda row: normalize(row['features'], target_scaler), axis=1)
                data['test']['X_target']['data']['features'] = data['test']['X_target']['data'].apply(lambda row: normalize(row['features'], target_scaler), axis=1)
                if verbose: print(('info: ' if print_mode=='dev' else '')+'Done')
            else:
                if verbose: print(('warning: ' if print_mode=='dev' else '')+'-- Requested '+scale_target_features+' scaling for the target features while supplying images. This step will be skipped')

    return data['train'], data['val'], data['test'], data['info']

class BaseDataset(Dataset):
    """A custom pytorch Dataset with a flexible implementation that can handle different cases of instance and target features. 
       The speed of this could be improved by splitting this logic into multiple datasets designed for specific cases. 
    """    
    def __init__(self, config, data, instance_features, target_features, instance_transform=None, target_transform=None):
        self.config = config
        self.instance_branch_input_dim = config['instance_branch_input_dim']
        self.target_branch_input_dim = config['target_branch_input_dim']

        self.use_instance_features = config['use_instance_features']
        self.use_target_features = config['use_target_features']

        if instance_transform is not None:
            self.instance_transform = instance_transform

        if target_transform is not None:
            self.target_transform = target_transform

        self.triplet_data = data['data']
        self.instance_features = None
        self.target_features = None
        if instance_features is not None:
            self.instance_features = instance_features['data']
        if target_features is not None:
            self.target_features = target_features['data']

    def __len__(self):
        return len(self.triplet_data)

    def __getitem__(self, idx): 
        instance_id = int(self.triplet_data.iloc[idx]['instance_id'])
        target_id = int(self.triplet_data.iloc[idx]['target_id'])
        value = self.triplet_data.iloc[idx]['value']
        instance_features_vec = None
        target_features_vec = None

        if self.instance_features is not None:
            if self.config['instance_branch_architecture'] == 'CONV':
                image = Image.open(self.instance_features.loc[instance_id, 'dir']).convert('RGB')
                instance_features_vec = self.instance_transform(image)
            else:
                instance_features_vec = self.instance_features.loc[instance_id, 'features']
        else:
            instance_features_vec = np.zeros(self.instance_branch_input_dim, dtype=np.float32)
            instance_features_vec[instance_id] = 1

        if self.target_features is not None:
            if self.config['target_branch_architecture'] == 'CONV':
                image = Image.open(self.target_features.loc[target_id, 'dir']).convert('RGB')
                target_features_vec = self.target_transform(image)
            else:
                target_features_vec = self.target_features.loc[target_id, 'features']
        else:
            target_features_vec = np.zeros(self.target_branch_input_dim, dtype=np.float32) 
            target_features_vec[target_id] = 1

        return{'instance_id': instance_id, 'target_id': target_id, 'instance_features': instance_features_vec, 'target_features': target_features_vec, 'score': value}


'''
class BaseDataset(Dataset):
    def __init__(self, config, data, instance_features, target_features, dyadic_features=None, transform=None):
        self.config = config
        self.instance_branch_input_dim = config['instance_branch_input_dim']
        self.target_branch_input_dim = config['target_branch_input_dim']

        self.use_instance_features = config['use_instance_features']
        self.use_target_features = config['use_target_features']

        self.triplet_data = data['data']
        self.instance_features = None
        self.target_features = None
        if instance_features is not None:
            self.instance_features = instance_features['data']
        if target_features is not None:
            self.target_features = target_features['data']

    def __len__(self):
        return len(self.triplet_data)

    def __getitem__(self, idx): 
        instance_id = int(self.triplet_data.iloc[idx]['instance_id'])
        target_id = int(self.triplet_data.iloc[idx]['target_id'])
        value = self.triplet_data.iloc[idx]['value']
        instance_features_vec = None
        target_features_vec = None

        if self.instance_features is not None:
            instance_features_vec = self.instance_features.loc[instance_id]['features']
        else:
            instance_features_vec = np.zeros(self.instance_branch_input_dim)
            instance_features_vec[instance_id] = 1

        if self.target_features is not None:
            target_features_vec = self.target_features.loc[target_id]['features']
        else:
            target_features_vec = np.zeros(self.target_branch_input_dim) 
            target_features_vec[target_id] = 1

        return{'instance_id': instance_id, 'target_id': target_id, 'instance_features': instance_features_vec, 'target_features': target_features_vec, 'score': value}
'''