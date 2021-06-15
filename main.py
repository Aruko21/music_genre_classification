import os
import sys
import sklearn as skl
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import audio_features as af
from multiprocessing import Pool

SAVE_FILE = True


def file_handle(filename):
    # filename = files[i]

    audio_proc = af.AudioProcessing(filename)
    audio_features = af.AudioFeatures(audio_proc)

    # features = audio_features.get_mfccs(n_mfcc=13, n_mels=88, delta=True, delta2=True, stat_funcs=("median", "square"))
    # features.update(audio_features.get_rhythm())
    features = audio_features.get_rhythm()
    features.update(audio_features.get_harmonic_change(stat_funcs=("median", "square",)))

    return features, filename


def generate_feature_set(csv_filename, genres, root):
    csv_writer = af.FeaturesCSVWriter(csv_filename)

    for genre in genres[:]:
        print("processing '{}' genre...".format(genre))

        files = af.get_audios_by_genre(root, genre)[:]

        pool = Pool()
        result = pool.map(file_handle, files)

        for entry in result:
            filename = entry[1]
            features = entry[0]
            csv_writer.append_audio(filename.split("/")[-1], features, genre)

    csv_writer.generate_csv()


def svm_classify(csv_features, plots_name="Genre classification", kernel="linear", random_state=None, c_param=1.0,
                 get_statistics=False, matrix_name="Confusion matrix SVM", save_file=False):
    ml_plots = af.MLPlots(name=plots_name, save_file=save_file)
    classifier = af.GenreClassifier(skl.svm.SVC(kernel=kernel, C=c_param), csv_features, random_state=random_state)

    classifier.train()
    y_pred = classifier.predict()
    f1_score = classifier.get_f1_score(y_pred)

    if get_statistics:
        classifier.print_report(y_pred)
        print(f1_score)
        ml_plots.plot_confusion_matrix(classifier, matrix_name)

    return classifier, f1_score


def svm_comparison(plots_name="Comparison svm"):
    ml_plots = af.MLPlots(name=plots_name)
    attempts = 100
    f1_scores = {
        "Stratified dataset": [],
        "Nonstratified dataset": []
    }

    for i in range(attempts):
        print("{} of {} attempts computing...".format(i + 1, attempts))
        classifier = af.GenreClassifier(skl.svm.SVC(kernel='linear'), "gtzan_features.csv")
        classifier.train()
        y_pred = classifier.predict()
        f1_scores["Stratified dataset"].append(classifier.get_f1_score(y_pred))

        classifier_withoutstr = af.GenreClassifier(skl.svm.SVC(kernel='linear'), "gtzan_features.csv", stratify=False)
        classifier_withoutstr.train()
        y_pred = classifier_withoutstr.predict()
        f1_scores["Nonstratified dataset"].append(classifier_withoutstr.get_f1_score(y_pred))

    ml_plots.plot_comparison(f1_scores, x_len=attempts, name="Data set splitting methods comparison")
    print("non stratified mean: ", np.mean(f1_scores["Nonstratified dataset"]))
    print("stratified mean: ", np.mean(f1_scores["Stratified dataset"]))


def get_optimize_c(features, kernel, rand_state):
    def optimize_c(c_param):
        _, score = svm_classify(features, kernel=kernel, random_state=rand_state, c_param=c_param)
        return 1 - score
    return optimize_c


def main():
    # af.sound_envelope_demo(save_file=SAVE_FILE)
    # af.piano_demo(save_file=SAVE_FILE)
    # af.genres_mfcc_demo(save_file=SAVE_FILE)

    gtzan_genres = af.get_all_directories("gtzan")

    csv_features = "gtzan_features"
    generate_feature_set("gtzan_features", genres=gtzan_genres, root="gtzan")
    # classifier, _ = svm_classify(csv_features, kernel="linear", random_state=10, c_param=0.1405)
    print("features computing done")

    print("calculate best linear SVM")
    optimize_linear = minimize_scalar(get_optimize_c(features=csv_features, kernel="linear", rand_state=10), bounds=(0.005, 1), method='bounded')
    linear_c = optimize_linear.x
    _, score = svm_classify(csv_features, kernel="linear", random_state=10, c_param=linear_c, get_statistics=True,
                            matrix_name="Confusion Matrix for linear SVM (BPM, harmonic change)", save_file=SAVE_FILE)
    print("Linear SVM | F1 = {}, optimized C param: {}".format(score, linear_c))

    print("calculate best RBF SVM")
    optimize_rbf = minimize_scalar(get_optimize_c(features=csv_features, kernel="rbf", rand_state=10),
                                   bounds=(1000, 10000), method='bounded')
    rbf_c = optimize_rbf.x
    _, score = svm_classify(csv_features, kernel="rbf", random_state=10, c_param=rbf_c, get_statistics=True,
                            matrix_name="Confusion Matrix for RBF SVM (BPM, harmonic change)", save_file=SAVE_FILE)
    print("RBF SVM | F1 = {}, optimized C param: {}".format(score, rbf_c))

    #
    # svm_comparison()


if __name__ == "__main__":
    main()
