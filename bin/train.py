# -*- coding: utf-8 -*-
import sys
import warnings

sys.path.append('./lib')
warnings.filterwarnings("ignore")

from logger import initialize_logger
import click
import optuna
import torch

from model.kfoldcv  import KFoldCV, \
                           ParallelKFoldCVStrategy, \
                           NonParallelKFoldCVStrategy

from optimizer import optimizer_sumary, \
                      ValAccPruneCallback

from fifa.dataset import FifaDataset

from fifa.model import train_model_1

from utils import set_device_name, \
                  get_device_name, \
                  get_device

from optuna.pruners import HyperbandPruner

def cv_strategy(k_fold):
    return ParallelKFoldCVStrategy(processes=k_fold) if 'cpu' == get_device_name() else NonParallelKFoldCVStrategy()

def objetive(trial, k_fold, X, y):
    cv = KFoldCV(
        model_train_fn = lambda train_set, val_set, params, callbacks, fold: train_model_1(train_set, val_set, params, callbacks, fold, trial),
        k_fold         = k_fold,
        callbacks      = [],
        strategy       = cv_strategy(k_fold)
    )

    return cv.train(
        X,
        y,
        params = {
            'hidden_layers': 1,
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

def setup_gpu(device, cuda_process_memory_fraction=0.2):
    if 'gpu' in device:
        torch.cuda.set_per_process_memory_fraction(
            cuda_process_memory_fraction, 
            get_device()
        )
        torch.cuda.empty_cache()

def load_dataset():
    dataset = FifaDataset(
        train_path = './tp2/dataset/fifa2021_training.csv',
        test_path  = './tp2/dataset/fifa2021_test.csv'
    )
    return dataset.train_features_target()

@click.command()
@click.option(
    '--device',
    default='cpu',
    help='Device used to train and optimize model. Values: gpu, cpu.'
)
@click.option('--study',  default='my-studio-10', help='The study name.')
@click.option('--trials',  default=200, help='Max trials count.')
@click.option(
    '--timeout',
    default=5000,
    help='maximum time spent optimizing hyper parameters in seconds.'
)
@click.option(
    '--db-url',
    default='mysql://root:1234@localhost/example',
    help='Mariadb/MySQL connection url.'
)
@click.option(
    '--cuda-process-memory-fraction',
    default=0.5,
    help='Setup max memory user per CUDA procees. Percentage expressed between 0 and 1'
)
@click.option('--folds',  default=5, help='Number of train dataset splits to apply cross validation.')
def main(device, study, trials, timeout, db_url, cuda_process_memory_fraction, folds):
    set_device_name(device)
    setup_gpu(device, cuda_process_memory_fraction)

    X, y = load_dataset()

    study_optimization = optuna.create_study(
        storage        = db_url,
        study_name     = study,
        load_if_exists = True,
        direction      = "maximize",
        pruner         = HyperbandPruner()
    )

    study_optimization.optimize(
        lambda trial: objetive(trial, folds, X, y),
        n_trials = trials,
        timeout  = timeout
    )

    optimizer_sumary(study_optimization)

if __name__ == '__main__':
    initialize_logger()
    main()
