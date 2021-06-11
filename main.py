import os
import sys
import sklearn as skl
import numpy as np
import pandas as pd
import audio_features as af
from multiprocessing import Pool

SAVE_FILE = True

MOCK_FILES = {
    "piano_sample": "audio/piano_pattern.wav",
    "piano_c": "audio/piano_c.wav",
    "violin_c": "audio/violin_c.wav",
    "gtzan_metal": "gtzan/metal/metal.00000.wav",
    "gtzan_pop": "gtzan/pop/pop.00000.wav",
    "gtzan_classical": "gtzan/classical/classical.00000.wav"
}


def sound_envelope_demo(piano_filepath, violin_filepath):
    piano_proc = af.AudioProcessing(piano_filepath)
    violin_proc = af.AudioProcessing(violin_filepath)

    piano_plots = af.AudioPlots(piano_proc, name="Piano C5", save_file=SAVE_FILE)
    violin_plots = af.AudioPlots(violin_proc, name="Violin C3", save_file=SAVE_FILE)
    piano_plots.plot_waveform()
    violin_plots.plot_waveform()



def piano_demo(filepath):
    # В librosa есть набор тестовых сигналов, доступ к которым осуществляется с помощью функции
    # librosa.ex("trumpet")

    piano_proc = af.AudioProcessing(filepath)
    piano_features = af.AudioFeatures(piano_proc)
    piano_plots = af.AudioPlots(piano_proc, name="piano sample", save_file=SAVE_FILE)

    piano_plots.plot_waveform()
    # vector shape is (sr * duration)
    print(
        "Shape of waveform with sample rate '{}' is: {}".format(piano_proc.get_sr(), piano_proc.get_signal().shape)
    )

    stft_matrix = piano_proc.get_stft(af.AudioProcessing.DEF_FRAME_SIZE, af.AudioProcessing.DEF_HOP_SIZE)
    print("The shape of wave after getting stft: '{}'. Type: {}".format(stft_matrix.shape, type(stft_matrix[0][0])))

    piano_plots.plot_power_spectrogram()

    # Получаем децибелы (меняем амплитуду)
    piano_plots.plot_db_spectrogram()

    piano_plots.mel_demo(n_mels=10)

    mel_spectrogram = piano_proc.get_mel_spec(n_mels=176)
    print("Mel spectrogram shape: ", mel_spectrogram.shape)

    piano_plots.plot_mel_spectrogram(n_mels=176)

    piano_plots.plot_mfcc(n_mfcc=13, n_mels=176)
    piano_plots.plot_mfcc(n_mfcc=13, n_mels=176, type="delta")
    piano_plots.plot_mfcc(n_mfcc=13, n_mels=176, type="delta2")

    features = piano_features.get_mfccs(n_mfcc=13, n_mels=176, delta=True, stat_funcs=("mean", "median", "square"))
    piano_plots.plot_mean_mfcc(features["mfcc_mean"], name="mean(MFCC's) for piano sample")

    return features


def file_handle(filename):
    # filename = files[i]

    audio_proc = af.AudioProcessing(filename)
    audio_features = af.AudioFeatures(audio_proc)

    features = audio_features.get_mfccs(n_mfcc=13, n_mels=88, delta=True, stat_funcs=("mean", "median", "square",))
    # features.update(audio_features.get_rhythm())
    # features.update(audio_features.get_harmonic_change(stat_funcs=("mean", "median", "square",)))

    return features, filename



def main():
    # sound_envelope_demo(MOCK_FILES.get("piano_c"), MOCK_FILES.get("violin_c"))
    # piano_features = piano_demo(MOCK_FILES.get("piano_sample"))

    # csv_writer = af.FeaturesCSVWriter("test_features")
    # csv_writer.append_audio(MOCK_FILES.get("piano_sample").split("/")[-1], piano_features, "piano")

    # gtzan_files = list(MOCK_FILES.values())[1:]
    # for file in gtzan_files:
    #     gtzan_proc = af.AudioProcessing(file)
    #     gtzan_features = af.AudioFeatures(gtzan_proc)
    #     gtzan_plots = af.AudioPlots(gtzan_proc, name=file.split("/")[-1], save_file=SAVE_FILE)
    #
    #     gtzan_plots.plot_mel_spectrogram(n_mels=176)
    #     gtzan_plots.plot_mfcc(n_mfcc=13, n_mels=176)
    #
    #     features = gtzan_features.get_mfccs(n_mfcc=13, n_mels=176, delta=True, stat_funcs=("mean", "median", "square"))
    #     csv_writer.append_audio(file.split("/")[-1], features, file.split("/")[1])
    #
    # csv_writer.generate_csv()

    gtzan_genres = af.get_all_directories("gtzan")

    csv_writer = af.FeaturesCSVWriter("gtzan_features")

    print("test: ", af.get_audios_by_genre("gtzan", "blues"))
    for genre in gtzan_genres[:]:
        print("processing '{}' genre...".format(genre))

        files = af.get_audios_by_genre("gtzan", genre)[:]

        pool = Pool()
        result = pool.map(file_handle, files)

        for entry in result:
            filename = entry[1]
            features = entry[0]
            csv_writer.append_audio(filename.split("/")[-1], features, genre)

    csv_writer.generate_csv()

    ml_plots = af.MLPlots(name="test_ml")
    classifier = af.GenreClassifier(skl.svm.SVC(kernel='linear'), "gtzan_features.csv", random_state=10)
    classifier.train()
    y_pred = classifier.predict()
    classifier.print_report(y_pred)
    print(classifier.get_f1_score(y_pred))
    ml_plots.plot_confusion_matrix(classifier, "What")

    # ml_plots = af.MLPlots(name="test_ml")
    # attempts = 100
    # f1_scores = {
    #     "Stratified dataset": [],
    #     "Nonstratified dataset": []
    # }
    #
    # for i in range(attempts):
    #     print("{} of {} attempts computing...".format(i + 1, attempts))
    #     classifier = af.GenreClassifier(skl.svm.SVC(kernel='linear'), "gtzan_features.csv")
    #     classifier.train()
    #     y_pred = classifier.predict()
    #     f1_scores["Stratified dataset"].append(classifier.get_f1_score(y_pred))
    #
    #     classifier_withoutstr = af.GenreClassifier(skl.svm.SVC(kernel='linear'), "gtzan_features.csv", stratify=False)
    #     classifier_withoutstr.train()
    #     y_pred = classifier_withoutstr.predict()
    #     f1_scores["Nonstratified dataset"].append(classifier_withoutstr.get_f1_score(y_pred))
    #
    # ml_plots.plot_comparison(f1_scores, x_len=attempts, name="Data set splitting methods comparison")
    # print("non stratified mean: ", np.mean(f1_scores["Nonstratified dataset"]))
    # print("stratified mean: ", np.mean(f1_scores["Stratified dataset"]))


if __name__ == "__main__":
    main()
