import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sklearn as skl

from .processing import AudioProcessing

WAVE_FIGSIZE = (12, 6)
SPECTRO_FIGSIZE = (8, 6)
DPI = 200
SAVE_IMG_LOCATION = "plots"


class AudioPlots:
    def __init__(self, features, dpi=200, save_file=False, name=None):
        self._audio = features
        self.dpi = dpi
        self.save_file = save_file
        self.name = name
        self.filename = "_".join(name.split("."))
        self.save_loc = SAVE_IMG_LOCATION

    def change_save_location(self, dirpath):
        self.save_loc = dirpath

    def plot_waveform(self, figsize=WAVE_FIGSIZE):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=self.dpi)
        if self.name:
            fig.suptitle("Waveform of '{}'".format(self.name))
        else:
            fig.suptitle("Waveform of music signal")

        librosa.display.waveplot(
            self._audio.get_signal(),
            sr=self._audio.get_sr(),
            ax=axes
        )
        axes.set_ylabel("Amplitude")

        plt.show()

        if self.save_file:
            fig.savefig(os.path.join(self.save_loc, "_".join(["waveform", self.filename])), dpi=self.dpi)

        return

    def plot_spectrogram(self, matrix, hop_size, spectype="linear", figsize=SPECTRO_FIGSIZE, colorbar_label=None,
                         colorbar_dimension="", name=None):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=self.dpi)
        if name:
            fig.suptitle("{} for {}".format(name, self.name))

        img = librosa.display.specshow(
            matrix,
            sr=self._audio.get_sr(),
            hop_length=hop_size,
            x_axis="time",
            y_axis=spectype,
            ax=axes
        )
        # legend
        cb = plt.colorbar(img, format="%+2.f {}".format(colorbar_dimension))
        if colorbar_label:
            cb.set_label(colorbar_label)
        plt.show()

        if self.save_file:
            fig.savefig(os.path.join(self.save_loc, "_".join(["spec", name, self.filename])), dpi=self.dpi)

    def plot_power_spectrogram(self, frame_size=AudioProcessing.DEF_FRAME_SIZE, hop_size=AudioProcessing.DEF_HOP_SIZE,
                               name="Power amplitude spectrogram"):
        spec = self._audio.get_stft(frame_size=frame_size, hop_size=hop_size)
        self.plot_spectrogram(spec, hop_size=hop_size, spectype="linear", name=name, colorbar_label="Amplitude")

    def plot_db_spectrogram(self, frame_size=AudioProcessing.DEF_FRAME_SIZE, hop_size=AudioProcessing.DEF_HOP_SIZE,
                            name="Db amplitude spectrogram"):
        spec = self._audio.get_db_amplitude(frame_size=frame_size, hop_size=hop_size)
        self.plot_spectrogram(spec, hop_size=hop_size, spectype="log", name=name, colorbar_dimension="dB")

    def mel_demo(self, n_mels, frame_size=AudioProcessing.DEF_FRAME_SIZE):
        # n_fft - number of frmaes
        filter_banks = librosa.filters.mel(n_fft=frame_size, sr=self._audio.get_sr(), n_mels=n_mels)
        # (mel_bands, 2048 / 2 + 1)
        print("mel_demo: filter banks shape: ", filter_banks.shape)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=WAVE_FIGSIZE, dpi=DPI)
        fig.suptitle("Mel-filter banks ({}) for {}".format(n_mels, self.name))
        img = librosa.display.specshow(
            filter_banks,
            sr=self._audio.get_sr(),
            x_axis="linear",
            ax=axes
        )
        plt.colorbar(img, format="%+2.f")
        plt.show()

        if self.save_file:
            fig.savefig(os.path.join(self.save_loc, "_".join(["mel-demo", str(n_mels)])), dpi=self.dpi)

    def plot_mel_spectrogram(self, n_mels, frame_size=AudioProcessing.DEF_FRAME_SIZE,
                             hop_size=AudioProcessing.DEF_HOP_SIZE, name="Mel-spectrogram"):
        spec = self._audio.get_mel_spec(n_mels=n_mels, frame_size=frame_size, hop_size=hop_size)
        self.plot_spectrogram(spec, hop_size=hop_size, spectype="mel", name=name, colorbar_dimension="dB")

    def plot_mfcc(self, n_mfcc, n_mels, type="default", frame_size=AudioProcessing.DEF_FRAME_SIZE,
                  hop_size=AudioProcessing.DEF_HOP_SIZE):
        mfcc = self._audio.get_mfcc(n_mfcc=n_mfcc, n_mels=n_mels, frame_size=frame_size, hop_size=hop_size)
        plot_name = "MFCC's"
        file_name = "mfcc"
        if type == "delta":
            mfcc = self._audio.get_mfcc_delta(mfcc)
            plot_name = r"$\Delta$ MFCC's"
            file_name = "dmfcc"
        elif type == "delta2":
            plot_name = r"$\Delta \Delta$ MFCC's"
            file_name = "ddmfcc"
            mfcc = self._audio.get_mfcc_delta2(mfcc)

        print("Test mfcc shape: ", mfcc.shape)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=WAVE_FIGSIZE, dpi=DPI)
        fig.suptitle("{} for {}".format(plot_name, self.name))

        img = librosa.display.specshow(
            mfcc,
            sr=self._audio.get_sr(),
            x_axis="time",
            ax=axes
        )
        plt.colorbar(img, format="%+2.f")
        plt.show()

        if self.save_file:
            fig.savefig(os.path.join(self.save_loc, "_".join([file_name, str(n_mfcc), self.filename])), dpi=self.dpi)

    def plot_mean_mfcc(self, mfccs, name):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=WAVE_FIGSIZE, dpi=DPI)
        fig.suptitle(name)

        axes.plot(np.arange(1, len(mfccs) + 1), mfccs, linestyle="-", marker='o')
        plt.show()

        if self.save_file:
            fig.savefig(os.path.join(self.save_loc, name), dpi=self.dpi)


class MLPlots:
    def __init__(self, dpi=200, save_file=False, name=None):
        self.dpi = dpi
        self.save_file = save_file
        self.name = name
        self.filename = "_".join(name.split("."))
        self.save_loc = SAVE_IMG_LOCATION

    def plot_confusion_matrix(self, genreclassifier, name):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=300)

        skl.metrics.plot_confusion_matrix(genreclassifier.classifier, genreclassifier.X_test, genreclassifier.y_test,
                                          cmap=plt.cm.get_cmap("Blues"), normalize="true", ax=axes)

        plt.show()

        if self.save_file:
            fig.savefig(os.path.join(self.save_loc, name), dpi=self.dpi)

    def plot_comparison(self, data, x_len, name):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=300)
        fig.suptitle(name)

        for label, y_data in data.items():
            axes.plot(np.arange(1, x_len + 1), y_data, linestyle="-", marker='o', label=label)

        axes.legend(loc='upper right')
        axes.set_ylabel("F1-Measure Score")
        axes.set_xlabel("Attempts")
        axes.grid()
        plt.show()

        if self.save_file:
            fig.savefig(os.path.join(self.save_loc, name), dpi=self.dpi)
