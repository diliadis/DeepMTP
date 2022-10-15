Loading a dataset
#################

Loading a built-in dataset
**************************
Loading one of the datasets offered natively by DeepMTP
In the example above, the multi-label classification dataset is loaded my one of the built-in functions offered by **DeepMTP**. More specifically the available functions are the following:


**load_process_MLC()**
======================

The user can load the multi-label classification datasets available in the `MULAN repository <https://mulan.sourceforge.net/datasets-mlc.html>`_. The different datasets can be accessed by specifying a valid name in the dataset_name parameter.::

    data_mlc = load_process_MLC(dataset_name='bibtex', variant='undivided', features_type='dataframe')
    train_mlc, val_mlc, test_mlc, data_info_mlc = data_process(data_mlc, validation_setting='B', verbose=True)


**load_process_MTR()**
======================

The user can load the multivariate regression datasets available in the `MULAN repository <https://mulan.sourceforge.net/datasets-mtr.html>`_. The different datasets can be accessed by specifying a valid name in the dataset_name parameter.::

    mtr_data = load_process_MTR(dataset_name='enb', features_type='dataframe')
    train_mtr, val_mtr, test_mtr, data_info_mtr = data_process(mtr_data, validation_setting='B', verbose=True)


**load_process_MTL()**
======================

The user can load the multi-task learning dataset dog, a crowdsourcing dataset first introduced in `Liu et al <https://ieeexplore.ieee.org/document/8440116/>`_. More specifically, the dataset contains 800 images of dogs who have been partially labelled by 52 annotators with one of 5 possible breeds. To modify this multi-class problem to a binary problem, we modify the task so that the prediction involves the correct or incorrect labelling by the annotator. In a future version of the software another dataset of the same type will be added.::

    train_mtl, val_mtl, test_mtl, data_info_mtl = data_process(mtl_data, validation_setting='B', verbose=True)
    train_mtl, val_mtl, test_mtl, data_info_mtl = data_process(mtl_data, validation_setting='B', verbose=True)


**load_process_MC()**
=====================

The user can load the matrix completion dataset MovieLens 100K, a movie rating prediction dataset available by the the `GroupLens lab <https://grouplens.org/datasets/movielens/>`_ that contains 100k ratings from 1000 users on 1700 movies. In a future version of the software larger versions of the movielens dataset will be added::
    
    dp_data = load_process_DP(dataset_name='ern')
    train_dp, val_dp, test_dp, data_info_dp = data_process(dp_data, validation_setting='D', verbose=True)


**load_process_DP()**
=====================

The user can load dyadic prediction datasets available `here <https://people.montefiore.uliege.be/schrynemackers/datasets>`_. These are four different biological network datasets (ern, srn, dpie, dpii) which can be accessed by specifying one of the four keywords in the dataset_name parameter.::

    mc_data = load_process_MC(dataset_name='ml-100k')
    train_mc, val_mc, test_mc, data_info_mc = data_process(mc_data, validation_setting='A', verbose=True)


Loading a custom dataset
************************
In the most abstract view of a multi-target prediction problem there are three at most datasets that can be needed. These include the interaction matrix, the instance features, and the target features. When accounting for a train, val, test split the total number raises to 9 possible data sources. To group this info and avoid passing 9 different parameters in the data_process function, the framework uses a single dictionary with 3 key-value pairs {'train':{}, 'val':{}, 'test':{}}. The values should also be a dictionaries with 3 key-value pairs {'y':{}, 'X_instance':{}, 'X_target':{}}. When combined the dictionary can have the following form: {'train':{}, 'val':{}, 'test':{}}::

    data = {
        'train': {
            'y': ,
            'X_instance': ,
            'X_target': ,
        },
        'val': {
            'y': ,
            'X_instance': ,
            'X_target': ,
        },
        'test': {
            'y': ,
            'X_instance': ,
            'X_target': ,
        },
    }
