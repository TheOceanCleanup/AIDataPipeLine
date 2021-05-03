# AIDAtaPipeLine - A series of examples and utilities for Azure Machine Learning Services
# Copyright (C) 2020-2021 The Ocean Cleanup™
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from sklearn import svm
import joblib
import numpy as np
from azureml.core import Run
import argparse


# get hold of the current run
run = Run.get_context()

# let user feed in 2 parameters, the dataset to mount or download, and the regularization rate of the logistic regression model
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')
args = parser.parse_args()

data_folder = args.data_folder
print('Data folder:', data_folder)

# The actual model created here does nothing with the provided data and
# parameters. It's just a little example
# customer ages
X_train = np.array([50, 17, 35, 23, 28, 40, 31, 29, 19, 62])
X_train = X_train.reshape(-1, 1)
# churn y/n
y_train = ["yes", "no", "no", "no", "yes", "yes", "yes", "no", "no", "yes"]

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X_train, y_train)

# Using run.log() you can provided metrics on the models' performance. We also
# return the parameter here, so that too is registered
run.log('regularization rate', np.float(args.reg))
run.log('accuracy', 0.97)

# Write the model file to the outputs/ folder. This is then automatically
# attached to the Run, and to any models registered from that run
joblib.dump(value=clf, filename="outputs/churn-model-2.pkl")

with open('logs/stdout.log', 'w') as f:
    f.write('Done!\n')
