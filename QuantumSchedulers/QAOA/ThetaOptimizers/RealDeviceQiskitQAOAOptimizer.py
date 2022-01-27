from QuantumSchedulers.QAOA.QAOA import ThetaOptimizer, QCSampler, CircuitBuilder
from QuantumSchedulers.QAOA.QCSamplers.LazySampler import LazySampler
from QuantumSchedulers.QAOA.ThetaOptimizers.QiskitMinimizer import expected_value
from QuantumSchedulers.QAOA.ThetaOptimizers.QiskitQAOAOptimizer import get_quadradic_program
from qiskit_optimization.runtime import QAOAClient
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np
import math


class RealDeviceQiskitQAOAOptimizer(ThetaOptimizer):
    def __init__(self, provider, device_name: str, method='SPSA'):
        super().__init__()
        self._method = method
        self._device_name = device_name
        self._provider = provider

    def optimize_theta(self, circuit_builder: CircuitBuilder, qc_sampler: LazySampler, num_reads: int, hamiltonian,
                       theta):
        self._circuit_builder = circuit_builder
        self._qc_sampler = qc_sampler
        self._num_reads = num_reads
        self._hamiltonian = hamiltonian

        backend = self._provider.get_backend(self._device_name)

        qubo = get_quadradic_program(0.5 * (self._hamiltonian + np.transpose(self._hamiltonian)))
        print(qubo)
        self._qaoa_mes = QAOAClient(initial_point=theta, reps=int(len(theta)/2),
                                    optimizer={'name': self._method}, backend=backend, provider=self._provider,
                                    shots=self._num_reads, measurement_error_mitigation=False)

        qaoa = MinimumEigenOptimizer(self._qaoa_mes)
        self._solution = qaoa.solve(qubo)

        self._theta = self._solution.min_eigen_solver_result.optimal_point
        counts = self._solution.min_eigen_solver_result.eigenstate
        sum = 0
        for key, prob in counts.items():
            #we multiply probabilites with num_reads, as they will be normed by num_reads_eval afterwards
            #therefore, we have to choose num_reads = num_reads_eval!
            counts[key] = self._num_reads * prob ** 2 #square of eigenstate magnitude is prob for result
            sum += counts[key]
        self._qc_sampler.set_counts(counts)
        #assert int(sum) == self._num_reads, "Probabilities do not match" sometimes doesnt work because of rounding
        self._expected_energy = expected_value(counts, self._num_reads, self._hamiltonian)

    def get_name(self):
        return "REALDEVICEQISKITQAOAOPTIMIZER_" + self._method

