import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt


class GenreClassifier:
    def __init__(self, classifier, csv_file=None, test_size=0.20, random_state=None, stratify=True):
        self.classifier = classifier
        self.labels = None

        features_filename = csv_file
        if csv_file is not None and csv_file.split(".")[-1] != "csv":
            features_filename += ".csv"

        self.csv_file = features_filename

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        if self.csv_file is not None:
            self.upload_data(self.csv_file, test_size, random_state, stratify)

    def upload_data(self, csv_file, test_size=0.20, random_state=None, stratify=True):
        headers = list(pd.read_csv(csv_file, nrows=1))
        data = pd.read_csv(csv_file, usecols=[entry for entry in headers if entry not in ("filename",)])
        audio_features = data.drop('label', axis=1)
        labels = data["label"]
        self.labels = labels

        if stratify:
            stratify_set = labels
        else:
            stratify_set = None

        self.X_train, self.X_test, self.y_train, self.y_test = skl.model_selection.train_test_split(
            audio_features, labels, test_size=test_size, random_state=random_state, stratify=stratify_set
        )

    def train(self):
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError("Upload data before training the model")
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.classifier.predict(self.X_test)
        return y_pred

    def print_report(self, y_pred):
        print(skl.metrics.classification_report(self.y_test, y_pred))

    def get_f1_score(self, y_pred):
        return skl.metrics.f1_score(self.y_test, y_pred, average='weighted')
