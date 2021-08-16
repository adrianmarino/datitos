import matplotlib.pyplot as plt
import seaborn as sns

def plot_losses(train_losses, val_losses, epochs, warmup_count = 0):
    sns.set_style("darkgrid")
    sns.lineplot(
        x=epochs[warmup_count:],
        y=train_losses[warmup_count:],
        label='Train'
    )
    sns.lineplot(
        x=epochs[warmup_count:],
        y=val_losses[warmup_count:], 
        label='Validation'
    )
    plt.xlabel("Epocs")
    plt.ylabel("Loss")
    plt.title("Train vs. Validation")
    plt.tight_layout()
    plt.show()