import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=11)

# Average CV score on the training set was:0.7307377647154933
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.5, max_iter=4000, tol=0.1)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_depth=5, max_features=0.6500000000000001, min_samples_leaf=0.20500000000000002, min_samples_split=0.30500000000000005, n_estimators=350)),
    MinMaxScaler(),
    LGBMRegressor(colsample_bytree=0.75, learning_rate=0.1, max_bin=31, max_depth=6, min_child_weight=15, n_estimators=350, num_leaves=128, objective="mse", reg_alpha=5, subsample=0.85, subsample_freq=0, verbosity=-1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
