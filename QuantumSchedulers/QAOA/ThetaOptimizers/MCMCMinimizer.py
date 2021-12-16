from QuantumSchedulers.QAOA.QAOA import ThetaOptimizer, QCSampler, CircuitBuilder
from QuantumSchedulers.QAOA.ThetaOptimizers.QiskitMinimizer import get_expectation, key_to_vector
from scipy.optimize import minimize
import numpy as np
from typing import Tuple, List
from scipy.stats import norm, uniform
from scipy.stats._continuous_distns import norm_gen
import math

class MCMCMinimizer(ThetaOptimizer):
    def __init__(self, method: str, beta=0.2, xi=0.7, epsilon=0.001, eta=0.001, mcmc_steps=1000):
        super().__init__()
        self._method = method
        self._beta = beta
        self._xi = xi
        self._epsilon = epsilon
        self._eta = eta
        self._mcmc_steps = mcmc_steps

    def optimize_theta(self, circuit_builder: CircuitBuilder, qc_sampler: QCSampler, num_reads: int, hamiltonian,
                       theta, verbose=False):
        self._circuit_builder = circuit_builder
        self._qc_sampler = qc_sampler
        self._num_reads = num_reads
        self._hamiltonian = hamiltonian

        for i in range(self._mcmc_steps):
            theta = self.mcmc_step(theta)

        expectation = get_expectation(self._hamiltonian, self._circuit_builder, self._qc_sampler, self._num_reads)
        bounds = [(0, math.pi) if k < theta.size/2 else (0, 2*math.pi) for k in range(theta.size)]
        res = minimize(expectation, theta, method=self._method, bounds=bounds)
        self._theta = res.x
        self._expected_energy = res.fun
        if verbose:
            print(res)

    def get_name(self):
        return "MCMC_QAOA"

    def mcmc_step(self, theta_t):
        lambda_t, sigma2_t = self.compute_total_lambda_and_sigma2(theta_t)
        grad_lambda_t = self.compute_grad_lambda(theta_t)
        #proposed new theta
        proposal_distributions_t = self.get_proposal_distributions(grad_lambda_t, sigma2_t)

        theta_d = np.array([dist.rvs() for dist in proposal_distributions_t])
        lambda_d, sigma2_d = self.compute_total_lambda_and_sigma2(theta_d)
        grad_lambda_d = self.compute_grad_lambda(theta_d)
        proposal_distributions_d = self.get_proposal_distributions(grad_lambda_d, sigma2_d)
        a = self.acceptance_distribution(theta_d, theta_t, lambda_d, lambda_t, proposal_distributions_d,
                                         proposal_distributions_d)
        u = uniform.rvs()
        theta_t_next = theta_t
        if a > u:
            theta_t_next = theta_d

        return theta_t_next

    def acceptance_distribution(self, theta_d: np.ndarray, theta_t: np.ndarray, lambda_d: float, lambda_t: float,
                                proposal_distributions_d: List[norm_gen], proposal_distributions_t: List[norm_gen]):

        g_d_given_t = np.prod([proposal_distributions_t[k].pdf(theta_t[k] - theta_d[k]) for k in range(theta_d.size)])
        g_t_given_d = np.prod([proposal_distributions_d[k].pdf(theta_d[k] - theta_t[k]) for k in range(theta_t.size)])
        return np.minimum(1, self.p_boltzmann(lambda_d) * g_t_given_d /(self.p_boltzmann(lambda_t) * g_d_given_t))

    def get_proposal_distributions(self, grad_lambda: np.ndarray, sigma2: float) -> List[norm_gen]:
        return [norm(self._eta * grad_lambda[k], self._xi**2 + self._eta**2 * sigma2 / (2 * self._epsilon**2))
                for k in range(grad_lambda.size)]

    def p_boltzmann(self, x: float):
        np.exp(-self._beta * x)

    def compute_total_lambda_and_sigma2(self, theta: np.ndarray) -> Tuple[float, float]:
        counts = self.execute_circ(theta)
        mu_raw = self.compute_mu_raw(counts)
        mu = self.compute_mu_weighted(mu_raw)
        sigma2 = self.compute_sigma2(mu_raw)
        return np.sum(mu), np.sum(sigma2)/self._num_reads

    def compute_grad_lambda(self, theta: np.ndarray) -> np.ndarray:
        return np.array([self.compute_grad_lambda_k(theta, k) for k in theta.size])

    def compute_grad_lambda_k(self, theta: np.ndarray, k: int):
        epsilon_k_hat = self._epsilon * np.eye(1, theta.size, k)
        return (self.compute_lambda(theta + epsilon_k_hat) - self.compute_lambda(theta - epsilon_k_hat)) \
               / (2 * self._epsilon)

    def compute_lambda(self, theta):
        counts = self.execute_circ(theta)
        mu_raw = self.compute_mu_raw(counts)
        mu = self.compute_mu_weighted(mu_raw)
        return np.sum(mu)

    def compute_mu_weighted(self, mu_raw) -> np.ndarray:
        return np.multiply(self._hamiltonian, mu_raw)

    def compute_mu_raw(self, counts: dict) -> np.ndarray:
        mu = np.zeros(self._hamiltonian.shape)

        for bitstring, count in counts.items():
            x = key_to_vector(bitstring)
            mu += count * np.matmul(x, x.transpose())
        return mu / self._num_reads

    def compute_sigma2(self, mu_raw: np.ndarray) -> np.ndarray:
        return np.multiply(np.multiply(self._hamiltonian, self._hamiltonian), np.ones(mu_raw.shape)
                           - np.multiply(mu_raw, mu_raw))

    def execute_circ(self, theta: np.ndarray) -> dict:
        qc = self._circuit_builder.get_quantum_circuit(theta)
        return self._qc_sampler.get_counts(qc, self._num_reads)
