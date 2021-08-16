import matplotlib.pyplot as plt
import seaborn as sns

def plot_losses(train_losses, val_losses, warmup_count = 0):
    sns.set_style("darkgrid")
    sns.lineplot(data=train_losses[warmup_count:], label='Train')
    sns.lineplot(data=val_losses[warmup_count:], label='Validation')
    plt.xlabel("Epocs")
    plt.ylabel("Loss")
    plt.title("Train vs. Validation")
    plt.tight_layout()
    plt.show()