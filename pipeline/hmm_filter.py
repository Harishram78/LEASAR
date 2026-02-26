"""
LEASAR - HMM Temporal State Filter
Hidden Markov Model for temporal anomaly state inference.
States: Normal (0) -> Degraded (1) -> Faulty (2)
M.Tech AI Thesis - Harish Ramachandran, SRM University 2025
"""

import numpy as np
from hmmlearn import hmm
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class HMMTemporalFilter:
    """
    HMM-based temporal state inference for LEASAR pipeline.

    Hidden States:
        0 = Normal     : Robot operating within safe parameters
        1 = Degraded   : Early warning - performance degradation detected
        2 = Faulty     : Critical anomaly - intervention required

    Observation:
        Discretised anomaly score from EnsembleAnomalyDetector (0-9 bins)
    """

    N_BINS = 10  # Discretisation bins for anomaly score

    def __init__(self, n_states: int = 3, n_iter: int = 100):
        self.n_states = n_states
        self.n_iter   = n_iter
        self.state_names = {0: 'NORMAL', 1: 'DEGRADED', 2: 'FAULTY'}

        # Transition matrix: mostly stays in current state
        self.trans_matrix = np.array([
            [0.90, 0.09, 0.01],   # Normal   -> Normal/Degraded/Faulty
            [0.10, 0.80, 0.10],   # Degraded -> Normal/Degraded/Faulty
            [0.05, 0.15, 0.80],   # Faulty   -> Normal/Degraded/Faulty
        ])

        # Emission: P(obs_bin | state)
        self.emission_matrix = np.array([
            [0.45, 0.30, 0.15, 0.05, 0.02, 0.01, 0.01, 0.00, 0.00, 0.01],  # Normal
            [0.10, 0.20, 0.25, 0.20, 0.12, 0.07, 0.03, 0.02, 0.01, 0.00],  # Degraded
            [0.01, 0.02, 0.05, 0.08, 0.12, 0.18, 0.22, 0.18, 0.10, 0.04],  # Faulty
        ])

        self.start_prob = np.array([0.90, 0.09, 0.01])

        self.model = hmm.CategoricalHMM(
            n_components=n_states,
            n_iter=n_iter,
            verbose=False
        )
        self.model.startprob_  = self.start_prob
        self.model.transmat_   = self.trans_matrix
        self.model.emissionprob_ = self.emission_matrix
        self._fitted = False
        self._history: List[int] = []   # Sliding window of recent states
        logger.info("HMMTemporalFilter initialised | states=%d | bins=%d", n_states, self.N_BINS)

    def _discretise(self, score: float) -> int:
        """Bin anomaly score [0,1] into integer observation [0, N_BINS-1]."""
        return int(min(score * self.N_BINS, self.N_BINS - 1))

    def fit(self, score_sequences: List[List[float]]):
        """
        Fit HMM on sequences of anomaly scores from labelled normal operation.
        Args:
            score_sequences: List of score sequences, each a list of floats in [0,1]
        """
        obs_seqs = [np.array([self._discretise(s) for s in seq]).reshape(-1, 1)
                    for seq in score_sequences]
        X   = np.vstack(obs_seqs)
        lengths = [len(s) for s in obs_seqs]
        self.model.fit(X, lengths)
        self._fitted = True
        logger.info("HMM fitted on %d sequences (%d total obs)", len(score_sequences), len(X))
        return self

    def filter(self, score: float) -> np.ndarray:
        """
        Online filtering: infer current state from single anomaly score.
        Uses the most likely state given the emission probabilities.
        Returns state vector [anomaly_score, P(Normal), P(Degraded), P(Faulty)]
        """
        obs_bin   = self._discretise(score)
        emission  = self.emission_matrix[:, obs_bin]
        state_probs = emission / emission.sum()
        current_state = int(np.argmax(state_probs))
        self._history.append(current_state)
        if len(self._history) > 100:
            self._history.pop(0)
        return np.array([score] + state_probs.tolist())

    def viterbi(self, scores: List[float]) -> Tuple[List[int], float]:
        """
        Viterbi decoding: most likely state sequence for a full score sequence.
        Returns: (state_sequence, log_likelihood)
        """
        obs = np.array([self._discretise(s) for s in scores]).reshape(-1, 1)
        log_likelihood, state_sequence = self.model.decode(obs, algorithm='viterbi')
        labels = [self.state_names[s] for s in state_sequence]
        return labels, log_likelihood

    def log_likelihood(self, scores: List[float]) -> float:
        """Score a sequence - low LL flags anomalous sequences."""
        obs = np.array([self._discretise(s) for s in scores]).reshape(-1, 1)
        return self.model.score(obs)

    @property
    def recent_state(self) -> str:
        """Return most recent inferred state name."""
        if not self._history:
            return 'NORMAL'
        return self.state_names.get(self._history[-1], 'UNKNOWN')

    @property
    def fault_rate(self) -> float:
        """Fraction of recent history in FAULTY state (window=100)."""
        if not self._history:
            return 0.0
        return sum(s == 2 for s in self._history) / len(self._history)


if __name__ == '__main__':
    np.random.seed(42)
    hmm_filter = HMMTemporalFilter(n_states=3)

    # Simulate: normal scores then spike to anomalous
    normal_scores  = np.random.beta(2, 8, 50).tolist()   # low anomaly
    anomaly_scores = np.random.beta(8, 2, 10).tolist()   # high anomaly
    sequence = normal_scores + anomaly_scores

    for i, score in enumerate(sequence):
        state_vec = hmm_filter.filter(score)
        if i % 10 == 0 or score > 0.7:
            print(f't={i:3d} | score={score:.3f} | state={hmm_filter.recent_state} | '
                  f'P(F)={state_vec[3]:.3f}')

    labels, ll = hmm_filter.viterbi(sequence)
    print(f'\nViterbi LL={ll:.2f} | Last 5 states: {labels[-5:]}')
    print(f'Fault rate (last 100): {hmm_filter.fault_rate:.2%}')
