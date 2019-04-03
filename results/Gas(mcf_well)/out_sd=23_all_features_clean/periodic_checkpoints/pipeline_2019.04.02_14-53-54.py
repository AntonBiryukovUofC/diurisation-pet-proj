import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=23)

# Average CV score on the training set was:0.6828527617647385
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LGBMRegressor(colsample_bytree=0.95, learning_rate=0.001, max_bin=255, max_depth=4, min_child_weight=10, n_estimators=100, num_leaves=128, objective="quantile", reg_alpha=30, subsample=0.85, subsample_freq=10, verbosity=-1)),
    LGBMRegressor(colsample_bytree=0.8, learning_rate=0.2, max_bin=31, max_depth=7, min_child_weight=20, n_estimators=50, num_leaves=16, objective="mse", reg_alpha=30, subsample=0.9, subsample_freq=30, verbosity=-1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
