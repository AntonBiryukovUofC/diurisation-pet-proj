import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=23)

# Average CV score on the training set was:0.7028253069533194
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_depth=8, max_features=0.05, min_samples_leaf=0.255, min_samples_split=0.005, n_estimators=200)),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_depth=6, max_features=0.05, min_samples_leaf=0.255, min_samples_split=0.255, n_estimators=450)),
    LGBMRegressor(colsample_bytree=0.65, learning_rate=0.05, max_bin=31, max_depth=7, min_child_weight=20, n_estimators=350, num_leaves=16, objective="mse", reg_alpha=5, subsample=0.55, subsample_freq=0, verbosity=-1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
