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
                         get_device_name, \
                         set_device_memory

from dict_utils   import dict_join
from file_utils   import create_dir

from optimizer import save_contour_plot, \
                      save_edf_plot, \
                      save_optimization_history_plot, \
                      save_parallel_coordinate_plot, \
                      save_param_importances_plot, \
                      save_slice_plot, \
                      save_trials_metric_dist_post, \
                      save_accurary_plot
# -----------------------------------------------------------------------------
#
#
#
#
#
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def cv_strategy(k_fold):
    return ParallelKFoldCVStrategy(processes=k_fold) if  get_device_name() == 'cpu' else NonParallelKFoldCVStrategy()

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
    create_dir(report_path)
 
    study = optuna.load_study(storage = db_url, study_name = study)
    optimizer_sumary(study)

    save_optimization_history_plot(study, report_path)
    save_parallel_coordinate_plot(study, report_path)
    save_param_importances_plot(study, report_path)
    save_slice_plot(study, report_path)
    save_contour_plot(study, report_path)
    save_edf_plot(study, report_path)
    save_trials_metric_dist_post(study, report_path)

    X, y = FifaDataset.load_train_features_target()
    accs = get_accuracy_dist(study, seeds_count, folds, X, y)

    print(accs)
    save_accurary_plot(accs, report_path)

if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------