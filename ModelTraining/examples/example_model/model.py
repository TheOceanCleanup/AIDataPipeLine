import pickle


class Model:
    def __init__(self, param_a, param_b):
        self.param_a = param_a
        self.param_b = param_b
        self.var = 0
        self.labels = []

    def train(self, weights, folder, labels):
        lbl_list = []

        with open(weights) as f:
            pass

        # Labels are provided per image, as a list of dicts in the column
        # labels
        for i, l in labels.iterrows():
            with open(folder + l['image_url'], 'rb') as f:
                lbl_list += [x['label'] for x in l['label']]

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
