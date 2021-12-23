# -*- coding: utf-8 -*-
import click
import sys
sys.path.append('./lib')

import warnings
warnings.filterwarnings("ignore")

from logger import initialize_logger

import numpy as np

# Pandas...
import pandas as pd

import optuna
import torch

from model.kfoldcv  import KFoldCV, \
                           ParallelKFoldCVStrategy, \
                           NonParallelKFoldCVStrategy

from callbacks import ReduceLROnPlateau

from optimizer import optimizer_sumary, \
                      ValAccPruneCallback, \
                      plot_trials_metric_dist

from fifa.dataset import FifaDataset

from fifa.model import FifaModel1, \
                       train_model_1

from utils import set_device_name, \
                  get_device_name

dataset = FifaDataset(
    train_path = './tp2/dataset/fifa2021_training.csv',
    test_path  = './tp2/dataset/fifa2021_test.csv'
)

X, y = dataset.train_features_target()

def objetive(trial):
    k_fold = 5

    strategy = ParallelKFoldCVStrategy(processes=k_fold) if 'cpu' == get_device_name() else NonParallelKFoldCVStrategy()

    cv = KFoldCV(
        model_train_fn = train_model_1, 
        k_fold         = 5,
        callbacks      = [ValAccPruneCallback(trial), ReduceLROnPlateau(patience=50)],
        strategy       = strategy
    )
    return cv.train(
        X,
        y,
        params = {
            'hidden_units':   trial.suggest_int  ('hidden_units',   10,   500,   step = 10  ),
            'lr':             trial.suggest_float('lr',             1e-7, 1e-1),
            'momentum':       trial.suggest_float('momentum',       0.01, 0.9,   step = 0.01),
            'dropout':        trial.suggest_float('dropout',        0.0,  0.4,   step = 0.01),
            'batch_size':     trial.suggest_int  ('batch_size',     256,  512,   step = 32  ),
            'epochs':         trial.suggest_int  ('epochs',         100,  2000,  step = 10  ),
            'seed':           trial.suggest_int  ('seed',           100,  1500,  step = 10  ),
            'relu_neg_slope': trial.suggest_float('relu_neg_slope', 0.0,  0.4,   step = 0.01)
        }
    )

@click.command()
@click.option('--device', default='cpu', help='gpu or cpu')
@click.option('--study',  default='my-studio-10', help='The study name.')
def main(device, study):
    set_device_name(device)

    study_optimization = optuna.create_study(
        storage="mysql://root:1234@localhost/example", 
        study_name=study,
        load_if_exists=True,
        direction="maximize"
    )

    study_optimization.optimize(
        objetive,
        n_trials = 200,
        timeout  = 5000
    )

    optimizer_sumary(study_optimization)


if __name__ == '__main__':
    initialize_logger()
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    torch.cuda.empty_cache()
    main()