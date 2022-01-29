from Scheduler import CPLEXSolver
from Reader import JobShopReader
from data import JobShopSchedulingData
import os
import matplotlib.pyplot as plt
import json
import numpy as np
from JobShopSampler import JobShopSampler


path = "js_example_files" + os.sep

sampler = JobShopSampler(path=path, nJobs=6, nMachines=5, pmin=1, pmax=8)
#sampler.sample(nsamples=100)
path_samples = path+"Samples6x5"+os.sep
solution = []
for filename in os.listdir(path_samples):
    reader = JobShopReader()
    reader.read_problem_data(path_samples+filename)
    data: JobShopSchedulingData = reader.get_data()

    solver = CPLEXSolver(data)
    solver.solve()
    solution.append(solver.get_plottable_solution())
    #solver.store_solution(path_samples+os.sep+"sol_"+filename)

solution = np.array(solution)
"""
reader = JobShopReader()
reader.read_problem_data(path + "mt06.txt")
data: JobShopSchedulingData = reader.get_data()

solver = CPLEXSolver(data)
solver.solve()


result_data = None
with open(path + "solution.txt") as json_file:
    result_data = json.load(json_file)

plottabl_sol = np.array(result_data["PLOTTABLE_SOLUTION"])
print(plottabl_sol[2,2])
"""


def f(x: float) -> int:
    return 0


