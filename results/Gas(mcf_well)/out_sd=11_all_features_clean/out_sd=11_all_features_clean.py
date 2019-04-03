import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=11)

# Average CV score on the training set was:0.7318608213857505
exported_pipeline = make_pipeline(
    StandardScaler(),
    MinMaxScaler(),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
    StandardScaler(),
    LGBMRegressor(colsample_bytree=0.5, learning_rate=0.1, max_bin=31, max_depth=6, min_child_weight=5, n_estimators=350, num_leaves=8, objective="mse", reg_alpha=0.0001, subsample=0.6, subsample_freq=0, verbosity=-1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
