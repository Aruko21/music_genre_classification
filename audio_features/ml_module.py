import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt


def handle_data(csv_file):
    cols = list(pd.read_csv(csv_file, nrows=1))
    data = pd.read_csv(csv_file, usecols=[i for i in cols if i not in ("filename",)])

    audio_features = data.drop('label', axis=1)
    labels = data["label"]

    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(audio_features, labels, test_size=0.20, random_state=18)

    svclassifier = skl.svm.SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    print(skl.metrics.plot_confusion_matrix(svclassifier, X_test, y_test))
    print(skl.metrics.classification_report(y_test, y_pred))
    plt.show()
