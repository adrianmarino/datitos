# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys
sys.path.append('./lib')

import warnings
warnings.filterwarnings("ignore")

import click
from logger import initialize_logger
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optuna
import torch

from model.kfoldcv  import KFoldCV, \
                           ParallelKFoldCVStrategy, \
                           NonParallelKFoldCVStrategy

from optimizer import optimizer_sumary, \
                      plot_trials_metric_dist

from fifa.model   import train_model_1
from fifa.dataset import FifaDataset

from device_utils import set_device_name, \
                         get_device, \
                         set_device_memory
from dict_utils   import dict_join
from file_utils   import create_dir

from plot import plot_hist, \
                 local_bin

from optuna.visualization import plot_contour, \
                                 plot_edf, \
                                 plot_optimization_history, \
                                 plot_parallel_coordinate, \
                                 plot_param_importances, \
                                 plot_slice
# -----------------------------------------------------------------------------
#
#
#
#
#
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def generate_plots(study, path, seeds_count, folds):
    create_dir(path)
 
    fig = plot_optimization_history(study)
    fig.update_layout(width=1000, height=500)
    fig.write_image(
        '{}/{}-optimization_history.png'.format(path, study.study_name),
        engine="kaleido"
    )
    
    fig = plot_parallel_coordinate(study)
    fig.write_image(
        '{}/{}-parallel_coordinate.png'.format(path, study.study_name), 
        engine="kaleido"
    )

    fig = plot_param_importances(study)
    fig.write_image(
        '{}/{}-param_importances.png'.format(path, study.study_name), 
        engine="kaleido"
    )

    fig = plot_slice(study)
    fig.write_image(
        '{}/{}-slice.png'.format(path, study.study_name), 
        engine="kaleido"
    )

    fig = plot_contour(study, params=["epochs", "lr"])
    fig.update_layout(width=1000, height=800)
    fig.write_image(
        '{}/{}-contour.png'.format(path, study.study_name), 
        engine="kaleido"
    )

    fig = plot_edf(study)
    fig.update_layout(width=500, height=500)
    fig.write_image(
        '{}/{}-edf.png'.format(path, study.study_name), 
        engine="kaleido"
    )

    plot_trials_metric_dist(study)
    plt.savefig('{}/{}-trials_metric_dist.png'.format(path, study.study_name))

    X, y = FifaDataset.load_train_features_target()

    accs = get_accuracy_dist(study, seeds_count, folds, X, y)
    print(accs)

    plot_hist(
        lambda: accs,
        bins_fn = local_bin(),
        xlabel  = 'Accuracy'
    )
    plt.savefig('{}/{}-acc_dist.png'.format(path, study.study_name))

def cv_strategy(k_fold):
    return ParallelKFoldCVStrategy(processes=k_fold) if 'cpu' == get_device_name() else NonParallelKFoldCVStrategy()

def get_accuracy_dist(study, seeds_count, folds, X, y):
    seeds             = random.sample(range(1,1000), seeds_count)
    best_hyper_params = study.best_trial.params.copy()
    accs              = []

    for seed in seeds:
        print('Seed:', seed)
        cv = KFoldCV(
            model_train_fn = train_model_1, 
            k_fold = folds, 
            strategy = cv_strategy(folds)
        )
        accs.append(cv.train(X, y, params = dict_join({ 'seed': seed, 'hidden_layers': 1 }, best_hyper_params)))
    
    return accs
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
    '--report-path',  
    default='./report', 
    help='Path where save optimization plots.'
)
@click.option(
    '--seeds-count',
    default=3,
    help='seeds count used calculate acuracy distribution'
)
@click.option(
    '--folds',  
    default=3, 
    help='Number of train dataset splits to apply cross validation.'
)
def main(device, study, db_url, report_path, seeds_count, folds):
    initialize_logger()
    set_device_name(device)
    set_device_memory(device)

    study = optuna.load_study(storage = db_url, study_name = study)
    optimizer_sumary(study)

    generate_plots(study, report_path, seeds_count, folds)

if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------