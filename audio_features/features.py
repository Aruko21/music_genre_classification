import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import csv

from .processing import AudioProcessing


class AudioFeatures:
    def __init__(self, processor):
        self.audio_proc = processor
        self.stat_funcs = {
            "mean": np.mean,
            "median": np.median,
            "square": np.std
        }

    def get_mfccs(self, n_mfcc, n_mels, frame_size=AudioProcessing.DEF_FRAME_SIZE,
                  hop_size=AudioProcessing.DEF_HOP_SIZE, delta=False, delta2=False, stat_funcs=("mean",)):
        mfccs = self.audio_proc.get_mfcc(n_mfcc=n_mfcc, n_mels=n_mels, frame_size=frame_size, hop_size=hop_size)
        if delta:
            mfccs_d = self.audio_proc.get_mfcc_delta(mfccs)
        if delta2:
            mfccs_d2 = self.audio_proc.get_mfcc_delta2(mfccs)

        result = {}
        for func in stat_funcs:
            stat_func = self.stat_funcs.get(func)
            if stat_func is None:
                raise ValueError("There is no '{}' statistic function".format(func))

            mfcc_key = "_".join(["mfcc", func])
            mfcc_d_key = "_".join(["mfcc_d", func])
            mfcc_d2_key = "_".join(["mfcc_d2", func])

            result[mfcc_key] = stat_func(mfccs, axis=1)
            if delta:
                result[mfcc_d_key] = stat_func(mfccs_d, axis=1)
            if delta2:
                result[mfcc_d2_key] = stat_func(mfccs_d2, axis=1)

        return result


class FeaturesCSVWriter:
    def __init__(self, filename):
        self.filename = filename
        self.header = ""
        self.rows = []

    def append_audio(self, filename, features, label):
        if self.header == "":
            self.header = "filename"
            for feature in features:
                if isinstance(features[feature], np.ndarray):
                    for i in range(len(features[feature])):
                        self.header += f" {feature}{i + 1}"
                else:
                    self.header += f" {feature}"
            self.header += " label"

        to_append = f"{filename}"
        for feature in features:
            if isinstance(features[feature], np.ndarray):
                for value in features[feature]:
                    to_append += f" {value}"
            else:
                to_append += f" {features[feature]}"

        to_append += f" {label}"
        self.rows.append(to_append)

    def generate_csv(self):
        file = open(f"{self.filename}.csv", "w")
        with file:
            writer = csv.writer(file)
            writer.writerow(self.header.split())
            for row in self.rows:
                writer.writerow(row.split())
