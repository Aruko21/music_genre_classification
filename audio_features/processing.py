import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


class AudioProcessing:
    # Размер окна, в котором рассчитывается ФФТ
    DEF_FRAME_SIZE = 2048
    # Размер перекрытия окон (Должно быть во избежание эффекта Гиббса)
    DEF_HOP_SIZE = 512

    def __init__(self, filepath, sample_rate=None, duration=None, offset=None):
        # sr=None for native sample rate from file
        # Если указать, или использовать значение по умолчанию (22050) - то произойдет resample
        signal, sr = librosa.load(path=filepath, mono=True, duration=duration, offset=offset, sr=sample_rate)
        self.filename = filepath.split("/")[-1]
        self._signal = signal
        self._sr = sr

        self._stft = None
        self._db_stft = None
        self._mel_spec = None
        self._mfcc = None

        self._prev_stft_frame = None
        self._prev_stft_hop = None
        self._prev_mel_frame = None
        self._prev_mel_hop = None

        self._stft_upd = False
        self._mel_upd = False

    def get_signal(self):
        return self._signal

    def get_sr(self):
        return self._sr

    def get_stft(self, frame_size=DEF_FRAME_SIZE, hop_size=DEF_HOP_SIZE):
        if self._stft is None or self._prev_stft_frame != frame_size or self._prev_stft_hop != hop_size:
            self._stft = librosa.stft(self._signal, n_fft=frame_size, hop_length=hop_size)
            # Т.к. изначально мы получили комплексные коэффициенты Фурье - для их визуализации необходимо перейти
            # к вещественным числам
            self._stft = np.abs(self._stft) ** 2

            self._prev_stft_frame = frame_size
            self._prev_stft_hop = hop_size
            self._stft_upd = True
        return self._stft

    def get_db_amplitude(self, frame_size=DEF_FRAME_SIZE, hop_size=DEF_HOP_SIZE):
        self._stft = self.get_stft(frame_size=frame_size, hop_size=hop_size)

        if self._db_stft is None or self._stft_upd is True:
            self._db_stft = librosa.power_to_db(self._stft)
            self._stft_upd = False

        return self._db_stft

    def get_mel_spec(self, n_mels, frame_size=DEF_FRAME_SIZE, hop_size=DEF_HOP_SIZE):
        if self._mel_spec is None or self._prev_mel_frame != frame_size or self._prev_mel_hop != hop_size:
            self._mel_spec = librosa.feature.melspectrogram(self._signal, sr=self._sr, n_fft=frame_size,
                                                            hop_length=hop_size, n_mels=n_mels, window="hamming")

            self._mel_spec = librosa.power_to_db(self._mel_spec)

            self._prev_mel_frame = frame_size
            self._prev_mel_hop = hop_size
            self._stft_upd = True
        return self._mel_spec

    def get_mfcc(self, n_mfcc, n_mels, frame_size=DEF_FRAME_SIZE, hop_size=DEF_HOP_SIZE):
        self._mel_spec = self.get_mel_spec(n_mels=n_mels, frame_size=frame_size, hop_size=hop_size)

        if self._mfcc is None or self._mel_upd is True:
            self._mfcc = librosa.feature.mfcc(S=self._mel_spec, n_mfcc=n_mfcc, sr=self._sr)
            self._mel_upd = False

        return self._mfcc

    def get_tempo(self):
        tempo = librosa.beat.tempo(self._signal, sr=self._sr)
        return tempo[0]

    def get_harmonics_change(self):
        harmonics_change = librosa.feature.tonnetz(self._signal, sr=self._sr)
        return harmonics_change

    @staticmethod
    def get_mfcc_delta(mfcc):
        return librosa.feature.delta(mfcc)

    @staticmethod
    def get_mfcc_delta2(mfcc):
        return librosa.feature.delta(mfcc, order=2)
