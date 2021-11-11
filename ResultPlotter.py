from Scheduler import JobShopScheduler
from Reader import JobShopReader
import json
import numpy as np


class ResultPlotter(JobShopScheduler):
    def __init__(self, filename):
        self._plottable_solution = None
        self._benchmarking_data = None
        self.solve(filename)
        self._reader = JobShopReader()
        self._reader.read_problem_data(self._benchmarking_data["PROBLEM_FILENAME"])
        self._data = self._reader.get_data()

        #gets plottable solution from filename
    def solve(self, filename):
        with open(filename) as json_file:
            self._benchmarking_data = json.load(json_file)
            self._plottable_solution = np.array(self._benchmarking_data["PLOTTABLE_SOLUTION"])


    def get_solver_name(self):
        return self._benchmarking_data["SOLVER_NAME"]

    def get_plottable_solution(self):
        return self._plottable_solution


