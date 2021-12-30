from optuna.visualization import plot_contour, \
                                 plot_edf, \
                                 plot_optimization_history, \
                                 plot_parallel_coordinate, \
                                 plot_param_importances, \
                                 plot_slice
from optimizer import plot_trials_metric_dist
import matplotlib.pyplot as plt
from plot import plot_hist, \
                 local_bin

def save_accurary_plot(study, path):
    plot_hist(
        lambda: accs,
        bins_fn = local_bin(),
        xlabel  = 'Accuracy'
    )
    plt.savefig('{}/{}-acc_dist.png'.format(path, study.study_name))

def save_trials_metric_dist_post(study, path):
    plot_trials_metric_dist(study)
    plt.savefig('{}/{}-trials_metric_dist.png'.format(path, study.study_name))

def save_optimization_history_plot(study, path, width=1000, height=500):
    fig = plot_optimization_history(study)
    fig.update_layout(width=width, height=height)
    fig.write_image(
        '{}/{}-optimization_history.png'.format(path, study.study_name),
        engine='kaleido'
    )

def save_parallel_coordinate_plot(study, path):
    fig = plot_parallel_coordinate(study)
    fig.write_image(
        '{}/{}-parallel_coordinate.png'.format(path, study.study_name), 
        engine='kaleido'
    )

def save_param_importances_plot(study, path):
    fig = plot_param_importances(study)
    fig.write_image(
        '{}/{}-param_importances.png'.format(path, study.study_name), 
        engine='kaleido'
    )

def save_slice_plot(study, path):
    fig = plot_slice(study)
    fig.write_image('{}/{}-slice.png'.format(path, study.study_name), engine='kaleido')
 
def save_contour_plot(study, path):
    fig = plot_contour(study, params=["epochs", "lr"])
    fig.update_layout(width=1000, height=800)
    fig.write_image('{}/{}-contour.png'.format(path, study.study_name), engine='kaleido')

def save_edf_plot(study, path, width=500, height=500):
    fig = plot_edf(study)
    fig.update_layout(width=width, height=height)
    fig.write_image('{}/{}-edf.png'.format(path, study.study_name), engine='kaleido')
