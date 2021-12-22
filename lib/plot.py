import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

DEFAULT_FIGURE_SIZE = (10,5)


def plot_metrics(logs, warmup_count = 0): 
    metric_names = logs.keys()
    epochs = logs['epoch'][warmup_count:]

    sns.set_style("darkgrid")
    for name in metric_names:
        if 'epoch' != name:
            sns.lineplot(
                x=epochs,
                y=logs[name][warmup_count:], 
                label=name.capitalize()
            )
    plt.xlabel("Epocs")
    plt.title("Metrics")
    plt.tight_layout()
    plt.show()
    
    
    
def linspace_bin(start=1, end=1, bin_size=20):
    return lambda values: np.linspace(
        start, 
        end, 
        len(values) if bin_size <= len(values) else bin_size
    )

def local_bin(bin_size=20):
    return lambda values: np.linspace(
        min(values), 
        max(values),
        len(values) if bin_size <= len(values) else bin_size
    )
    
def plot_hist(
    get_data_fn     = None,
    figsize         = DEFAULT_FIGURE_SIZE,
    ylabel          = 'Frecuencia',
    xlabel          = 'x',
    title           = '',
    bins_fn         = linspace_bin(),
    density         = True,
    title_font_size = 16,
    xmetric_offset  = 0.05,
    ymetric_offset  = 0.08,
    box_color       = 'lightblue',
    violin_color    = 'lightgreen',
    decimals        = 4
):    
    f, (ax_box, ax_violin, ax_hist) = plt.subplots(
        3, 
        sharex=True, 
        gridspec_kw={"height_ratios": (.15, .25, .75)}
    )
    
    ax_box.set_title(
        title if title else '{} Histogram'.format(xlabel), 
        fontsize = title_font_size
    )

    f.set_size_inches(figsize[0], figsize[1])

    values = get_data_fn()
    bins   = bins_fn(values)

    sns.boxplot(values, ax=ax_box, color=box_color)

    sns.violinplot(
        values, 
        ax=ax_violin, 
        color=violin_color,
        split=True,
    )
        
    if density:
        sns.distplot(values, hist=True, bins=bins, ax=ax_hist)
    else:
        sns.histplot(values, bins=bins, ax=ax_hist)

    ax_hist.axvline(
        x     = np.mean(values),
        color = 'blue',
        ls    = '--',
        lw    = 2.5
    )
    ax_hist.axvline(
        x     = np.median(values),
        color = 'red',
        ls    = '--',
        lw    = 2.5
    )
 
    mode_results = stats.mode(values)
    
    print(mode_results)
    
    for mode in mode_results.mode: 
        ax_hist.axvline(
            x     = mode, 
            color = 'green',
            ls    = '--',
            lw    = 2.5
        )

    labels = [
        'Density', 
        'Mean {}'.format(round(np.mean(values), decimals)),
        'Median {}'.format(round(np.median(values), decimals))
    ]
    
    for (mode, count) in zip(mode_results.mode, mode_results.count): 
        labels.append('Mode {} ({})'.format(round(mode, decimals), count))

    ax_hist.legend(
        loc='lower center', 
        labels=labels,
        bbox_to_anchor=(0.5, -0.7)
    )
    
    ax_hist.set_ylabel(ylabel)
    ax_hist.set_xlabel(xlabel)
