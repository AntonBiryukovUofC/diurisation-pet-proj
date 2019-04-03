import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=1010)

# Average CV score on the training set was:0.683573884377932
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
    StackingEstimator(estimator=LGBMRegressor(colsample_bytree=0.65, learning_rate=0.005, max_bin=127, max_depth=6, min_child_weight=0.5, n_estimators=100, num_leaves=2, objective="quantile", reg_alpha=10, subsample=0.7, subsample_freq=20, verbosity=-1)),
    LGBMRegressor(colsample_bytree=0.8, learning_rate=0.2, max_bin=127, max_depth=4, min_child_weight=10, n_estimators=150, num_leaves=32, objective="mae", reg_alpha=1.0, subsample=1.0, subsample_freq=5, verbosity=-1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
