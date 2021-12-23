import pandas as pd

def columns_with_missing(df): return [col for col in df.columns if df[col].isnull().any()]

class FifaDataset:

    INDIVIDUAL_FEATURES = [
        'Overal',
        'Potential',
        'Height',
        'Weight',
        'PreferredFoot',
        'Age',
        'PlayerWorkRate',
        'WeakFoot',
        'SkillMoves', 
        'Value',
        'Wage',
        'Club',
        'Club_KitNumber',
        'Club_JoinedClub',
        'Club_ContractLength'
    ]

    SKILL_FEATURES = [
        'BallControl', 'Dribbling', 'Marking', 'SlideTackle', 'StandTackle', 'Aggression',
        'Reactions', 'Interceptions', 'Vision', 'Composure', 'Crossing', 'ShortPass',
        'LongPass', 'Acceleration', 'Stamina', 'Strength', 'Balance', 'SprintSpeed',
        'Agility', 'Jumping', 'Heading', 'ShotPower', 'Finishing', 'LongShots',
        'Curve', 'FKAcc', 'Penalties', 'Volleys', 'GKDiving', 'GKHandling',
        'GKKicking', 'GKReflexes'
    ]
    TARGET = 'Position'

    CAL_COLS = [ 'PreferredFoot', 'PlayerWorkRate']

    def __init__(
        self, 
        train_path = './dataset/fifa2021_training.csv',
        test_path  = './dataset/fifa2021_test.csv'
    ):
        self.__train_set = pd.read_csv(train_path)
        self.__test_set  = pd.read_csv(test_path)

    def __feature_columns(self):
        return list(set(self.INDIVIDUAL_FEATURES + self.SKILL_FEATURES) -  set(columns_with_missing(self.__train_set)))

    def train_set(self):
        return self.__train_set[self.__feature_columns() + [self.TARGET]]

    def train_features_target(self):
        return self.__preprocess(self.train_set().dropna())

    def test_features(self):
        features, _ = self.__preprocess(self.__test_set)
        return features

    def test_set(self):
        return self.__test_set
        
    def __preprocess(self, df):
        features = df[self.__feature_columns()]

        # Tranform categorical columns to one-hot encodings..
        for col in self.CAL_COLS:
            features = pd.concat([features, pd.get_dummies(features[col], dummy_na=False)], axis=1)
        features = features.drop(self.CAL_COLS, axis=1)
        target   = pd.get_dummies(df[[self.TARGET]], dummy_na=False) if self.TARGET in df.columns else None

        return features, target