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

import pickle


class Model:
    def __init__(self, param_a, param_b):
        self.param_a = param_a
        self.param_b = param_b
        self.var = 0
        self.labels = []

    def train(self, dataset_file, label_file):
        lbl_list = []

        with open(dataset_file) as f:
            for l in f.readlines():
                l = l.rstrip('\n')
                filename = l.split(' ')[0]
                labels = l.split(' ')[1:]
                with open(filename, 'rb') as f2:
                    lbl_list += [x.split(',')[-1] for x in labels]

        self.labels = lbl_list

    def predict(self, file_path):
        idx = int(len(file_path) * self.param_a / self.param_b) % len(self.labels)
        i = self.labels[idx]
        return i

    def save(self, path):
        with open(path, 'wb') as f:
            return pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
