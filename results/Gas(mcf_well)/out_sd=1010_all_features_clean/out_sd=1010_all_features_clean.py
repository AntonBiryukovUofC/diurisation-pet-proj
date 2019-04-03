import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=1010)

# Average CV score on the training set was:0.7194892808201676
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
    StackingEstimator(estimator=RidgeCV()),
    RobustScaler(),
    LGBMRegressor(colsample_bytree=0.9, learning_rate=0.1, max_bin=63, max_depth=5, min_child_weight=10, n_estimators=300, num_leaves=8, objective="mse", reg_alpha=0, subsample=1.0, subsample_freq=5, verbosity=-1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
