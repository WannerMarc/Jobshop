from abc import ABCMeta, abstractmethod
from data import JobShopSchedulingData
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import json


class JobShopScheduler(metaclass=ABCMeta):
    def __init__(self, data: JobShopSchedulingData):
        self._data = data
        self._plottable_solution = None
        self._benchmarking_data = {"PLOTTABLE_SOLUTION": "NONE"}

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def get_plottable_solution(self):
        pass

    @abstractmethod
    def get_solver_name(self):
        pass

    def plot_solution(self):
        if self._plottable_solution is None:
            self.get_plottable_solution()
            print(self._plottable_solution)
        J, m, T = self._plottable_solution.shape
        fig, ax = plt.subplots()
        ax.set_title(self.get_solver_name())
        plt.xlim([0, T])
        plt.ylim([0, m])
        plt.yticks(np.arange(m)+np.ones(m)*0.25, ["Machine " + str(i) for i in range(m)])
        colorlist = []
        random.seed(1)
        for k in range(J):
            colorlist.append((random.random(), random.random(), random.random(), 0.5))
        legend_elements = [patches.Patch(facecolor=colorlist[i], label="Job " + str(i)) for i in
                           range(self._plottable_solution.shape[0])]
        for i in range(J):
            for o in range(m):
                t = 0
                while t < T:
                    if self._plottable_solution[i, o, t] == 1:
                        name = "Job: " + str(i) + " Op: " + str(o)
                        start = t
                        width = 1
                        while t + 1 < T and self._plottable_solution[i, o, t+1] == 1:
                            width += 1
                            t += 1

                        ax.add_artist(patches.Rectangle(xy=(start, self._data.get_M[i, o]), width=width, height=0.5,
                                                        fc=colorlist[i]))
                        ax.annotate(name, (start + 0.1, self._data.get_M[i, o] + 0.1), fontsize=6)
                    t += 1
        ax.legend(handles=legend_elements)

    def store_solution(self, filepath):
        self.get_plottable_solution()
        self._benchmarking_data["PLOTTABLE_SOLUTION"] = self._plottable_solution.tolist()
        print(self._benchmarking_data)
        with open(filepath, 'w+') as fp:
            json.dump(self._benchmarking_data, fp)

    def get_data(self) -> JobShopSchedulingData:
        return self._data

    def get_benchmarking_data(self):
        return self._benchmarking_data

from CPLEXScheduler import CPLEXSolver
from ResultPlotter import ResultPlotter