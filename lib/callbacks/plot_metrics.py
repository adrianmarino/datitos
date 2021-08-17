from plot import plot_metrics
from callbacks import OutputCallback, MetricLogger
from IPython.display import clear_output

class PlotMetrics(OutputCallback):
    def __init__(
        self, 
        warmup_count       = 0, 
        plot_each_n_epochs = 50, 
        reg_each_n_epochs  = 10, 
        metrics            = []
    ):
        super().__init__(plot_each_n_epochs)
        self.logger = MetricLogger()
        self.warmup_count  = warmup_count
        self.reg_each_n_epochs = reg_each_n_epochs
        self.metrics = metrics + ['epoch']

    def on_after_train(self, ctx):
        super().on_after_train(ctx)
        if ctx.epoch() % self.reg_each_n_epochs == 0: 
            [self.logger.append(metric, ctx.prop(metric)) for metric in self.metrics]

    def on_show(self, ctx):
        clear_output(wait=True)
        plot_metrics(self.logger.logs, self.warmup_count)