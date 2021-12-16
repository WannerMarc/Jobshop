from QuantumSchedulers.QAOA.QAOA import QCSampler
from qiskit import IBMQ
from qiskit import transpile, assemble
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor


class IBMRealDevice(QCSampler):
    def __init__(self, provider, device_name: str):
        super().__init__()
        self._device_name = device_name
        self._provider = provider

    def sample_qc(self, quantum_circuit, num_reads):
        backend = self._provider.get_backend(self._device_name)
        TQAOA = transpile(quantum_circuit, backend)
        qobj = assemble(TQAOA, shots=num_reads)
        job_exp = backend.run(qobj)
        job_monitor(job_exp)
        exp_results = job_exp.result()
        return exp_results.get_counts()

    def get_name(self):
        return "IBMREAL_DEVICE_"+self._device_name
