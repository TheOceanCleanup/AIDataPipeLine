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
        with open(file_path, 'rb') as f:
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
