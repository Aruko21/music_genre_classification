import os
import sys
import pandas as pd
from audio_features import *

SAVE_FILE=True


# В librosa есть набор тестовых сигналов, доступ к которым осущетвляется с помощью функции
# librosa.ex("trumpet")
piano_file = "audio/piano_pattern.wav"
gtzan_metal = "gtzan/metal/metal.00000.wav"
gtzan_pop = "gtzan/pop/pop.00000.wav"
gtzan_classical = "gtzan/classical/classical.00000.wav"

GTZAN = (gtzan_metal, gtzan_pop, gtzan_classical)

piano_proc = AudioProcessing(piano_file)
piano_features = AudioFeatures(piano_proc)
piano_plots = AudioPlots(piano_proc, name="piano sample", save_file=SAVE_FILE)
csv_writer = FeaturesCSVWriter("test_features")

piano_plots.plot_waveform()
# vector shape is (sr * duration)
print(
    "Shape of waveform with sample rate '{}' is: {}".format(piano_proc.get_sr(), piano_proc.get_signal().shape)
)

stft_matrix = piano_proc.get_stft(AudioProcessing.DEF_FRAME_SIZE, AudioProcessing.DEF_HOP_SIZE)
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
csv_writer.append_audio(piano_file.split("/")[-1], features, "piano")

for file in GTZAN:
    gtzan_proc = AudioProcessing(file)
    gtzan_features = AudioFeatures(gtzan_proc)
    gtzan_plots = AudioPlots(gtzan_proc, name=file.split("/")[-1], save_file=SAVE_FILE)

    gtzan_plots.plot_mel_spectrogram(n_mels=176)
    gtzan_plots.plot_mfcc(n_mfcc=13, n_mels=176)

    features = gtzan_features.get_mfccs(n_mfcc=13, n_mels=176, delta=True, stat_funcs=("mean", "median", "square"))
    csv_writer.append_audio(file.split("/")[-1], features, file.split("/")[1])

csv_writer.generate_csv()

data = pd.read_csv("test_features.csv")
print(data)

