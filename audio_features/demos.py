import audio_features as af
import os
from .utils import *


DEMO_FILES = {
    "piano_sample": "audio/piano_pattern.wav",
    "piano_c": "audio/piano_c.wav",
    "violin_c": "audio/violin_c.wav",
}


def sound_envelope_demo(save_file=False):
    piano_filepath = DEMO_FILES.get("piano_c")
    violin_filepath = DEMO_FILES.get("violin_c")

    for entry in (piano_filepath, violin_filepath):
        if not os.path.isfile(piano_filepath):
            raise ValueError("Cannot open '{}' file. Check your 'audio' directory".format(entry))

    piano_proc = af.AudioProcessing(piano_filepath)
    violin_proc = af.AudioProcessing(violin_filepath)

    piano_plots = af.AudioPlots(piano_proc, name="Piano C5", save_file=save_file)
    violin_plots = af.AudioPlots(violin_proc, name="Violin C3", save_file=save_file)
    piano_plots.plot_waveform()
    violin_plots.plot_waveform()


def piano_demo(save_file=False):
    # В librosa есть набор тестовых сигналов, доступ к которым осуществляется с помощью функции
    # librosa.ex("trumpet")

    filepath = DEMO_FILES.get("piano_sample")
    if not os.path.isfile(filepath):
        raise ValueError("Cannot open '{}' file. Check your 'audio' directory".format(filepath))

    piano_proc = af.AudioProcessing(filepath)
    piano_features = af.AudioFeatures(piano_proc)
    piano_plots = af.AudioPlots(piano_proc, name="piano sample", save_file=save_file)

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

    csv_writer = af.FeaturesCSVWriter("piano_sample_features")
    csv_writer.append_audio(filepath.split("/")[-1], piano_features, "piano")


def genres_mfcc_demo(save_file=False):
    root = "gtzan"
    genres = ("classical", "pop", "metal", "rock", "jazz", "disco")

    for genre in genres:
        filepath = get_audios_by_genre(root, genre)[0]

        genre_proc = af.AudioProcessing(filepath)
        genre_plots = af.AudioPlots(genre_proc, name="'{}' genre".format(genre), save_file=save_file)

        genre_plots.plot_mel_spectrogram(n_mels=176)
        genre_plots.plot_mfcc(n_mfcc=13, n_mels=176)

