# -*- coding: utf-8 -*-
import sys
import warnings

sys.path.append('./lib')
warnings.filterwarnings("ignore")

import click
from datetime import datetime
from logger import initialize_logger
import numpy as np
import pandas as pd
import optuna
import torch
from data         import to_single_col_df
from fifa.dataset import FifaDataset
from fifa.model   import FifaModel1
from utils import set_device_name, get_device
from file_utils import create_dir


def setup_gpu(device, cuda_process_memory_fraction=0.2):
    if 'gpu' in device:
        torch.cuda.set_per_process_memory_fraction(
            cuda_process_memory_fraction, 
            get_device()
        )
        torch.cuda.empty_cache()

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
        epochs       = params['epochs']
    )

    return model

def save_result(result_path, study, y_pred, dataset):
    mapping = ['DEF', 'FWD', 'GK', 'MID']
    test_data = pd.DataFrame(data={
        'ID': dataset.test_set().ID.values,
        'Category': [mapping[x] for x in y_pred.values.reshape((-1, ))]
    })

    create_dir(result_path)
    filename = "{}/{}-predict-{:%Y-%m-%d_%H-%M-%S}.csv".format(result_path, study.study_name, datetime.now())
    test_data.to_csv(filename, index=False)


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
def main(device, study, db_url, result_path):
    set_device_name(device)
    setup_gpu(device)

    study = optuna.load_study(storage = db_url, study_name = study)

    dataset = FifaDataset(
        train_path = './tp2/dataset/fifa2021_training.csv',
        test_path  = './tp2/dataset/fifa2021_test.csv'
    )
    X, y = dataset.train_features_target()

    model = train_model(X, y, params = study.best_trial.params)

    y_pred = to_single_col_df(model.predict(dataset.test_features()))

    save_result(result_path, study, y_pred, dataset)

if __name__ == '__main__':
    initialize_logger()
    main()
