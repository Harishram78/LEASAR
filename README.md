# LEASAR — Learning Enhanced Automation Systems with Adaptive Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch) ![License](https://img.shields.io/badge/License-Apache%202.0-green) ![Accuracy](https://img.shields.io/badge/Accuracy-97.1%25-brightgreen) ![F1](https://img.shields.io/badge/F1--Score-95.0%25-brightgreen) ![SRM University](https://img.shields.io/badge/M.Tech%20AI-SRM%20University%202025-orange)

> **M.Tech Artificial Intelligence Thesis — SRM University, Chennai, India (2025)**
> Harish Ramachandran | [LinkedIn](https://linkedin.com/in/harish-r-2bb95581)

---

## Overview

LEASAR addresses a critical gap in industrial robotics: traditional rule-based anomaly detection systems are brittle, require extensive manual tuning, and cannot adapt to novel fault conditions. This project develops a **hybrid ensemble anomaly detection pipeline** fused with an **adaptive reinforcement learning (RL) safety layer** for real-time monitoring and autonomous corrective action on industrial robots (UR3 and 6-DOF arm).

### Key Results

| Detector | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Isolation Forest | 89.3% | 87.1% | 91.2% | 89.1% |
| One-Class SVM | 85.7% | 83.4% | 88.6% | 85.9% |
| Autoencoder (AE) | 92.4% | 90.8% | 94.1% | 92.4% |
| LSTM Autoencoder | 94.8% | 93.2% | 96.4% | 94.8% |
| Statistical (Z-Score) | 81.2% | 79.6% | 83.1% | 81.3% |
| Rule-Based | 87.6% | 91.4% | 84.2% | 87.7% |
| **Ensemble (LEASAR)** | **97.1%** | **96.3%** | **97.9%** | **95.0%** |

### RL Algorithm Comparison

| Algorithm | Sample Efficiency | Noise Robustness | Avg Power (W) |
|---|---|---|---|
| DQN | 1.0x (baseline) | 78% | 24.1 |
| PPO | 1.4x | 85% | 22.3 |
| DDPG | 1.6x | 88% | 21.7 |
| **SAC** | **1.8x** | **92%** | **18.2** |

---

## Architecture

```
UR3 Robot (6-DOF)
|
+-- Joint Torques (J1-J6) ----+
+-- TCP Position (XYZ/RPY) ---+--> Preprocessing --> Kalman Filter --> Feature Eng.
+-- Force/Torque Sensor ------+
+-- Vision (YOLOv8) ----------+
 |
 v
+--------------------------------------------------+
| ENSEMBLE ANOMALY DETECTION LAYER |
| |
| [Isolation Forest] [One-Class SVM] |
| [Autoencoder] [LSTM-AE] |
| [Statistical] [Rule-Based] |
| |
| Majority Voting --> Anomaly Score (0-1) |
+--------------------------------------------------+
 |
 v
+--------------------------------------------------+
| HMM TEMPORAL FILTER |
| Normal --> Degraded --> Faulty |
| Viterbi Decoding | Forward Algorithm |
+--------------------------------------------------+
 |
 v
+--------------------------------------------------+
| SAC REINFORCEMENT LEARNING SAFETY LAYER |
| State: [anomaly_score, joint_states, tcp_pos] |
| Actions: Normal | ReduceSpeed | HALT | Alert |
| Reward: -anomaly_score - energy_cost |
+--------------------------------------------------+
 |
 v
 Safety Intervention / Robot Controller
```

---

## Repository Structure

```
LEASAR/
|
+-- models/
| +-- isolation_forest.py # Isolation Forest detector
| +-- autoencoder.py # PyTorch Autoencoder
| +-- lstm_autoencoder.py # LSTM Autoencoder for temporal detection
| +-- rule_based_detector.py # Hard safety threshold rules
|
+-- pipeline/
| +-- ensemble_detector.py # Master ensemble (majority voting)
| +-- preprocessing.py # Kalman filter + feature engineering
| +-- data_acquisition.py # UR3 sensor stream + OPC-UA
| +-- hmm_filter.py # HMM temporal state filter
|
+-- rl/
| +-- dqn_agent.py # DQN Agent with Replay Memory
| +-- sac_agent.py # SAC Agent (Twin Critic + Auto-entropy)
|
+-- deployment/
| +-- metrics_exporter.py # Prometheus metrics server
| +-- grafana_dashboard.json # Grafana dashboard (8 panels)
|
+-- notebooks/
| +-- 01_anomaly_detection_demo.ipynb # End-to-end demo notebook
|
+-- results/ # Plots, confusion matrices, metrics
+-- requirements.txt
+-- setup.py
+-- README.md
+-- LICENSE
```

---

## Installation

```bash
git clone https://github.com/Harishram78/LEASAR.git
cd LEASAR
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.0+
- scikit-learn, numpy, pandas
- stable-baselines3
- hmmlearn
- prometheus_client

---

## Quick Start

```python
from pipeline.data_acquisition import UR3DataAcquisition
from pipeline.preprocessing import RobotDataPreprocessor
from pipeline.ensemble_detector import EnsembleAnomalyDetector
from pipeline.hmm_filter import HMMTemporalFilter
from rl.sac_agent import SACAgent

# Initialize pipeline
dao = UR3DataAcquisition()
preprocessor = RobotDataPreprocessor()
detector = EnsembleAnomalyDetector()
hmm = HMMTemporalFilter(n_states=3)
agent = SACAgent(state_dim=14, action_dim=5)

# Run real-time loop
for sensor_data in dao.stream():
 features = preprocessor.transform(sensor_data)
 anomaly_score = detector.predict(features)
 state = hmm.filter(anomaly_score)
 action = agent.select_action(state)
 print(f"Action: {action} | Anomaly Score: {anomaly_score:.3f}")
```

---

## Research Alignment — LiU ELLIIT PhD Project

LEASAR directly addresses both research questions of the Linköping University ELLIIT project *"Learning Robot Control from Expert Demonstrations and Self-Experience"*:

**Q1 — Multi-Modal Expert Demonstrations:**
LEASAR fuses vision (YOLOv8), proprioception (XYZ/RPY joint states), and force/torque (J1-J6 torques) into a coherent ensemble detection pipeline, achieving 97.1% accuracy. The HMM temporal filter learns coherent state transition sequences from multi-modal sensor streams — directly analogous to skill acquisition from expert demonstrations.

**Q2 — Experiential Learning Beyond Demonstrations:**
The SAC agent with automatic entropy tuning handles continuous 6-DOF control and adapts to novel fault conditions unseen during training. SAC achieves 1.8x sample efficiency vs PPO and 92% noise robustness, enabling robust adaptation to task variations beyond demonstrated conditions.

---

## Citation

```bibtex
@mastersthesis{ramachandran2025leasar,
 title = {LEASAR: Learning Enhanced Automation Systems with Adaptive Reinforcement Learning},
 author = {Harish Ramachandran},
 school = {SRM University},
 year = {2025},
 address = {Chennai, India},
 note = {M.Tech Thesis, Department of Artificial Intelligence}
}
```

---

## License

Apache License 2.0 — See [LICENSE](LICENSE) for details.

---

*Developed as part of M.Tech AI research at SRM University, Chennai, India (2025)*
