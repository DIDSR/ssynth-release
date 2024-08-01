import argparse
import copy
import json
import os
import time

import h5py
import numpy as np
import progressbar
from PIL import Image
from scipy import ndimage


class skinLesion():
    settings = {
        "origProbabilities": [
            # inwards plane
            0.1, 0.1, 0.1,
            0.1, 0.5, 0.1,
            0.1, 0.1, 0.1,

            # same plane
            0.3, 0.4, 0.3,
            0.4, 0.1, 0.4,
            0.3, 0.4, 0.3,

            # outwards plane
            0.000, 0.0005, 0.000,
            0.0005, 0.002, 0.0005,
            0.000, 0.0005, 0.000,

            # additional 32 growing (+2 step)
            # in, out, up,
            # down, left, right
            0.5, 0.0001, 0.1,
            0.1, 0.1, 0.1
        ],
        "stepRange": (1, 2),  # growing step
        "gaussianSmooth": 0,
        "probabilityChangeStd": 0.3,
        # standrad deviation of the Gaussian distribution used to update the probabalities at each time point
        # "probabilityCancerA": 0.0001,
        "probabilityCancerP": 0.001,
        "cancerIterations": 10,
        "maxCancerRecursion": 3,
        "Niter": 25,  # number of iterations (i.e., end timepoint for the lesion to grow)
        "saveIterationsPerMinute": 999999,  # minutes
        "saveIterations": 5  # time step at which the lesion models are saved (#lesions = Niter/saveIterations)
    }

    directions = [
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),  # inwards plane
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),

        (0, -1, -1), (0, -1, 0), (0, -1, 1),  # same plane
        (0, 0, -1), (0, 0, 0), (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),

        (1, -1, -1), (1, -1, 0), (1, -1, 1),  # outwards plane
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),

        # 32 growing directions
        # in, out, up,
        # down, left, right
        (-2, 0, 0), (2, 0, 0), (0, -2, 0),
        (0, 2, 0), (0, 0, -2), (0, 0, 2)
    ]

    timepoint = 0

    def growCancer(self, cells, model, isCancer=0):
        oldcells = copy.deepcopy(cells)

        # change probabilities
        for idx in range(len(self.settings["origProbabilities"])):
            # do not change probability for outwards direction
            if self.directions[idx][0] <= 0:
                self.settings["origProbabilities"][idx] = max(0.01, min(0.99, self.settings["origProbabilities"][idx] *
                                                                        (1 + np.random.randn() * self.settings[
                                                                            "probabilityChangeStd"])))

        for idc, c in enumerate(oldcells):
            if time.time() > self.initTime and self.timepoint > 0:
                self.saveImage(model)
                self.timepoint += 1
                self.initTime = time.time() + 60 * \
                                self.settings["saveIterationsPerMinute"]

            # not isCancer an
            if (isCancer < self.settings["maxCancerRecursion"] and np.random.rand() < self.probabilityCancer):
                cancerCells = set([c])
                for _ in range(self.settings["cancerIterations"]):
                    model, cancerCells = self.growCancer(
                        cancerCells, model, isCancer + 1)
                cells.update(cancerCells)
            # if model[c[0], c[1], c[2]] > 0:
            #     continueT

            model[c[0], c[1], c[2]] = 255
            growingDirections = np.random.rand(
                len(self.settings["origProbabilities"]))
            idx = [x for x in range(len(growingDirections))]
            np.random.shuffle(idx)
            stepSize = np.random.randint(
                self.settings["stepRange"][0], self.settings["stepRange"][1] + 1)
            for dir in idx:
                if growingDirections[dir] < self.settings["origProbabilities"][dir]:
                    newcell = (c[0] + self.directions[dir][0] * stepSize,
                               c[1] + self.directions[dir][1] * stepSize,
                               c[2] + self.directions[dir][2] * stepSize)
                    #
                    if not (np.any(np.array(newcell) <= 0) or np.any(np.array(newcell) >= self.size - 1) or model[
                        newcell[0], newcell[1], newcell[2]] > 0):
                        surrounded = True
                        for row in (-1, 0, 1):
                            for col in (-1, 0, 1):
                                for dep in (-1, 0, 1):
                                    if model[row + c[0], col + c[1], dep + c[2]] == 0:
                                        surrounded = False
                                        break
                        if not surrounded:
                            cells.add((newcell))
                        # break

        oldcells = copy.deepcopy(cells)
        cells = set()

        for c in oldcells:
            surrounded = True
            for row in (-1, 0, 1):
                for col in (-1, 0, 1):
                    for dep in (-1, 0, 1):
                        # print(model[row + c[0], col + c[1], dep + c[2]])
                        if model[row + c[0], col + c[1], dep + c[2]] == 0:
                            surrounded = False
                            break
            if not surrounded:
                cells.add((c))

        return model, cells

    def saveImage(self, model):
        if self.settings["gaussianSmooth"] != 0:
            model = ndimage.gaussian_filter(
                model, sigma=self.settings["gaussianSmooth"])

        os.makedirs(os.path.dirname(__file__) +
                    f"/{self.outputFolder}/sample{self.seed}/T{self.timepoint + 1:03d}/", exist_ok=True)
        for idx, slice in enumerate(model):
            Image.fromarray(model[idx, :, :]).convert("L").save(
                os.path.dirname(__file__) + "/{:}/sample{:}/T{:03d}/slice{:03d}.png".format(self.outputFolder,
                                                                                            self.seed,
                                                                                            self.timepoint + 1, idx))

        norm = np.mean(model, axis=0)
        norm = 255 * (norm - np.min(norm)) / (np.max(norm) - np.min(norm))

        skinImage = np.empty(
            (norm.shape[0], norm.shape[1], 3), dtype=np.uint8)

        for row in range(norm.shape[0]):
            for col in range(norm.shape[1]):
                if not np.isnan(norm[row, col]):
                    skinImage[row, col] = (np.interp(
                        norm[row, col], [0, 255], [self.colorMap[0][0], self.colorMap[1][0]]),
                                           np.interp(
                                               norm[row, col], [0, 255], [self.colorMap[0][1], self.colorMap[1][1]]),
                                           np.interp(
                                               norm[row, col], [0, 255], [self.colorMap[0][2], self.colorMap[1][2]]))

        Image.fromarray(skinImage).save(os.path.dirname(
            __file__) + f"/{self.outputFolder}/PNG/sample{self.seed}T{self.timepoint + 1:03d}.png")

    def __init__(self, seed, saveDir, size=501):

        self.probabilityCancer = self.settings["probabilityCancerP"]
        self.outputFolder = saveDir + "/lesion_slices/"

        os.makedirs(os.path.dirname(__file__) +
                    f"/{self.outputFolder}", exist_ok=True)
        os.makedirs(os.path.dirname(__file__) +
                    f"/{self.outputFolder}/PNG", exist_ok=True)

        with open(os.path.dirname(__file__) + f"/{self.outputFolder}/settings.json", "w") as f:
            f.write(json.dumps(self.settings, indent=4))

        maxVolume = np.inf  # 200000

        self.size = size

        self.colorMap = [(223, 170, 139), (84, 51, 21)]
        self.initTime = time.time()
        self.timepoint = 0

        self.seed = seed

        np.random.seed(self.seed)
        model = np.zeros((size, size, size))

        cells = set([(size // 2, size // 2, size // 2)])
        bar = progressbar.ProgressBar(max_value=self.settings["Niter"])

        os.makedirs(os.path.dirname(__file__) +
                    f"/{self.outputFolder}/sample{self.seed}/", exist_ok=True)
        if os.path.exists(os.path.dirname(__file__) +
                          f"/{self.outputFolder}/sample{self.seed}/sample{self.seed}.h5"):
            os.remove(os.path.dirname(__file__) +
                      f"/{self.outputFolder}/sample{self.seed}/sample{self.seed}.h5")

        self.timepoint = 0

        for i in range(self.settings["Niter"]):
            bar.update(i)

            model, cells = self.growCancer(cells, model)

            if (self.timepoint + 1) % self.settings["saveIterations"] == 0:
                with h5py.File(
                        os.path.dirname(__file__) + f"/{self.outputFolder}/sample{self.seed}/sample{self.seed}.h5",
                        'a') as hf:
                    hflesion = hf.create_dataset(f"T{self.timepoint + 1:03d}", data=np.array(
                        model).astype(np.uint8), compression="gzip", compression_opts=9, track_times=False)

                self.saveImage(model)
            self.timepoint += 1

            if np.sum(model) / 255 > maxVolume:
                break

        bar.finish()


if __name__ == "__main__":
    size = 501

    parser = argparse.ArgumentParser()
    parser.add_argument('--lesion_ID', type=int, help='lesion ID', required='True')
    parser.add_argument('--saveDir', type=str, help='where to save outputs', default='../../data/outputs/')

    args = parser.parse_args()
    lesion_ID = args.lesion_ID
    if lesion_ID is None:
        lesion_ID = 100
    lesion_ID = int(lesion_ID)
    print("Lesion ID:", lesion_ID)
    lesion = skinLesion(lesion_ID, args.saveDir)
