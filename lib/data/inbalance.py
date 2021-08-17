import pandas as pd
from imblearn.over_sampling    import SMOTEN
from imblearn.under_sampling   import RandomUnderSampler
from imblearn.pipeline         import Pipeline


class OverUnderSampler:
    """
    Se encarga de realizar un over/under sampling sobre un dataframe de features. 
    Esto nos permite balancear el dataset especificando el grado:

        - over sampling: Sobre la clase minoritaria (entre 0 y 1).
        - under sampling: Sobre la clase mayoritaria (entre 0 y 1).
    """
    def __init__(
        self,
        random_state=None, 
        oversampling_strategy='auto', 
        undersampling_strategy='auto'
    ):
        oversampler = SMOTEN(
            random_state = random_state,
            sampling_strategy = undersampling_strategy
        )
        undersampler = RandomUnderSampler(sampling_strategy = undersampling_strategy)
        self.pipeline = Pipeline(steps=[('oversampler', oversampler), ('undersampler', undersampler)])
        
    def perform(self, features, target):
        bal_features, bal_target = self.pipeline.fit_resample(features.values, target.values)
        
        # Matrix to DataFrame
        bal_features = pd.DataFrame(data = bal_features, columns = features.columns)
        bal_target   = pd.DataFrame(data = bal_target,   columns = target.columns)

        return bal_features, bal_target