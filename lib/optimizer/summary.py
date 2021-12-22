import matplotlib.pyplot as plt
from plot import plot_hist, local_bin

from optuna.trial import TrialState

def optimizer_sumary(study):
    pruned_trials   = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

def plot_trials_metric_dist(study):
    plot_hist(
        lambda: [t.value for t in study.get_trials(states=[TrialState.COMPLETE])],    
        title   = 'Trials Accuracy Histogram',
        bins_fn = local_bin()
    )