# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys
import warnings

sys.path.append('./lib')
warnings.filterwarnings("ignore")

import logging
import click
from datetime import datetime
from logger import initialize_logger
import numpy as np
import pandas as pd
import optuna
from data         import to_single_col_df

from optimizer import optimizer_sumary
                      
from fifa.dataset import FifaDataset
from fifa.model   import FifaModel1

from callbacks    import CallbackSet, \
                         Logger, \
                         TrainValLossComputer, \
                         TrainValAccuracyComputer

from device_utils import set_device_name, \
                         get_device, \
                         set_device_memory
from dict_utils   import dict_join
from file_utils   import create_dir
# -----------------------------------------------------------------------------
#
#
#
#
#
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def train_model(X, y, params):
    np.random.seed(params['seed'])
    
    units_per_layer = \
        [X.shape[1]] + \
        [params['hidden_units'] for _ in range(params['hidden_layers'])] + \
        [y.shape[1]]

    model = FifaModel1(
        units_per_layer = units_per_layer,
        lr              = params['lr'],
        momentum        = params['momentum'],
        dropout         = params['dropout'],
        negative_slope  = params['relu_neg_slope']
    )

    model.fit(
        train_set    = (X, y), 
        val_set      = None, 
        batch_size   = params['batch_size'],
        epochs       = params['epochs'],
        callback_set = CallbackSet([
            TrainValLossComputer(),
            TrainValAccuracyComputer(),
            Logger(metrics=['epoch', 'train_acc'])
        ])
    )

    return model

def save_result(result_path, study, y_pred, dataset):
    mapping = ['DEF', 'FWD', 'GK', 'MID']
    test_data = pd.DataFrame(data={
        'ID': dataset.raw_test_set().ID.values,
        'Category': [mapping[x] for x in y_pred.values.reshape((-1, ))]
    })

    create_dir(result_path)
    filename = '{}/{}-predict-{:%Y-%m-%d_%H-%M-%S}.csv'.format(result_path, study.study_name, datetime.now())
    test_data.to_csv(filename, index=False)
    logging.info('{} file saved!'.format(filename))
# -----------------------------------------------------------------------------
#
#
#
#
#
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@click.command()
@click.option(
    '--device',
    default='cpu', 
    help='Device used to train and optimize model. Values: gpu, cpu.'
)
@click.option('--study',  default='my-studio-10', help='The study name.')
@click.option(
    '--db-url',  
    default='mysql://root:1234@localhost/example', 
    help='Mariadb/MySQL connection url.'
)
@click.option(
    '--result-path',
    default='./results',
    help='path where test predictions are saved.'
)
@click.option(
    '--cuda-process-memory-fraction',
    default=0.5,
    help='Setup max memory user per CUDA procees. Percentage expressed between 0 and 1'
)
def main(device, study, db_url, result_path, cuda_process_memory_fraction):
    initialize_logger()
    set_device_name(device)
    set_device_memory(device, cuda_process_memory_fraction)
 
    study = optuna.load_study(storage = db_url, study_name = study)

    dataset = FifaDataset()
    X, y = dataset.train_features_target()

    hyper_params = dict_join(study.best_trial.params, {'hidden_layers': 1})

    logging.info('Best hyper parameters:')
    optimizer_sumary(study)

    logging.info('Begin training...')
    model = train_model(X, y, params = hyper_params)

    y_pred = to_single_col_df(model.predict(dataset.test_features()))

    save_result(result_path, study, y_pred, dataset)

if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------