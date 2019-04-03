import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=11)

# Average CV score on the training set was:0.6855864709227676
exported_pipeline = LGBMRegressor(colsample_bytree=0.75, learning_rate=0.1, max_bin=31, max_depth=6, min_child_weight=5, n_estimators=350, num_leaves=8, objective="mae", reg_alpha=5, subsample=0.85, subsample_freq=20, verbosity=-1)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
