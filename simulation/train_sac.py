"""
LEASAR -- SAC Training Loop
============================
Trains the Soft Actor-Critic (SAC) agent (rl/sac_agent.py) inside the
UR3 PyBullet simulation environment (simulation/ur3_sim_env.py).

Features
--------
- Warm-up phase: random exploration before policy updates begin
- Episode-level logging: reward, fault counts, critic/actor losses
- Periodic checkpoint saves (actor + critic weights)
- CSV training log written to results/sac_training_log.csv
- Eval mode: run a fixed number of greedy episodes for final assessment
- Gymnasium/Gym dual compatibility (mirrors ur3_sim_env.py)

Usage
-----
  # Train from scratch (500 episodes, checkpoint every 50)
  python -m simulation.train_sac

  # Custom run
  python -m simulation.train_sac \
      --episodes 1000 \
      --fault_prob 0.2 \
      --save_dir checkpoints/run1 \
      --eval_episodes 10

Author  : Harish Ramachandran
Inst    : SRM Institute of Science and Technology, Chennai
License : Apache 2.0
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Resolve package imports whether run as a module or as a script
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # .../LEASAR/simulation
_ROOT = _HERE.parent                              # .../LEASAR
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.ur3_sim_env import UR3SimEnv, OBS_DIM, ACT_DIM
from rl.sac_agent import SACAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap_reset(result) -> np.ndarray:
    """Handle gymnasium (obs, info) or legacy gym obs."""
    if isinstance(result, tuple):
        return result[0]
    return result


def _unwrap_step(result) -> Tuple[np.ndarray, float, bool, dict]:
    """Return (obs, reward, done, info) regardless of gym version."""
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, reward, done, info = result
    return obs, float(reward), done, info


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    episodes: int = 500,
    max_steps: int = 500,
    fault_prob: float = 0.15,
    warmup_steps: int = 1000,
    save_dir: str = "checkpoints",
    save_every: int = 50,
    log_path: str = "results/sac_training_log.csv",
    eval_episodes: int = 5,
    render: bool = False,
    seed: int = 42,
) -> List[dict]:
    """
    Run SAC training inside UR3SimEnv and return episode statistics.

    Parameters
    ----------
    episodes      : Total training episodes.
    max_steps     : Max steps per episode (passed to UR3SimEnv).
    fault_prob    : Probability of injecting a fault per step.
    warmup_steps  : Steps of random exploration before policy updates.
    save_dir      : Directory for model checkpoints.
    save_every    : Save checkpoint every N episodes.
    log_path      : CSV file path for training metrics.
    eval_episodes : Number of greedy evaluation episodes run after training.
    render        : Open PyBullet GUI (slow, for debugging).
    seed          : Global RNG seed.
    """
    np.random.seed(seed)
    _ensure_dir(save_dir)
    _ensure_dir(str(Path(log_path).parent))

    # -- Environment ---------------------------------------------------------
    render_mode = "human" if render else None
    env = UR3SimEnv(
        render_mode=render_mode,
        fault_prob=fault_prob,
        max_steps=max_steps,
    )

    # -- Agent ---------------------------------------------------------------
    agent = SACAgent(
        state_dim=OBS_DIM,   # 24
        action_dim=ACT_DIM,  # 6
    )

    # -- Logging -------------------------------------------------------------
    csv_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "episode", "steps", "total_reward", "mean_reward",
            "fault_steps", "fault_rate",
            "critic_loss", "actor_loss", "alpha",
            "elapsed_s",
        ],
    )
    writer.writeheader()

    history: List[dict] = []
    global_step = 0
    t0 = time.time()

    print(
        f"\nLEASAR SAC Training"
        f"\n  Episodes   : {episodes}"
        f"\n  Max steps  : {max_steps}"
        f"\n  Fault prob : {fault_prob}"
        f"\n  Warmup     : {warmup_steps} steps"
        f"\n  Save dir   : {save_dir}"
        f"\n  OBS_DIM={OBS_DIM}  ACT_DIM={ACT_DIM}"
        f"\n{'=' * 55}"
    )

    # -- Main training loop --------------------------------------------------
    for ep in range(1, episodes + 1):
        obs = _unwrap_reset(env.reset(seed=seed + ep))
        ep_reward = 0.0
        ep_steps = 0
        fault_steps = 0
        losses: List[dict] = []

        while True:
            # Action selection: random during warm-up, policy thereafter
            if global_step < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, done, info = _unwrap_step(env.step(action))

            # Store transition (done=False for truncation to allow bootstrapping)
            is_terminal = info.get("fault", "none") == "collision" and done
            agent.store(obs, action, reward, next_obs, float(is_terminal))

            # Update agent
            if global_step >= warmup_steps:
                loss_info = agent.update()
                if loss_info:
                    losses.append(loss_info)

            # Track stats
            ep_reward += reward
            ep_steps += 1
            global_step += 1
            if info.get("fault", "none") != "none":
                fault_steps += 1

            obs = next_obs
            if done:
                break

        # -- Per-episode metrics --------------------------------------------
        mean_reward = ep_reward / max(ep_steps, 1)
        fault_rate = fault_steps / max(ep_steps, 1)
        avg_critic = float(np.mean([l["critic_loss"] for l in losses])) if losses else 0.0
        avg_actor = float(np.mean([l["actor_loss"] for l in losses])) if losses else 0.0
        avg_alpha = float(np.mean([l["alpha"] for l in losses])) if losses else 0.0
        elapsed = time.time() - t0

        row = {
            "episode":      ep,
            "steps":        ep_steps,
            "total_reward": round(ep_reward, 4),
            "mean_reward":  round(mean_reward, 4),
            "fault_steps":  fault_steps,
            "fault_rate":   round(fault_rate, 4),
            "critic_loss":  round(avg_critic, 6),
            "actor_loss":   round(avg_actor, 6),
            "alpha":        round(avg_alpha, 6),
            "elapsed_s":    round(elapsed, 1),
        }
        writer.writerow(row)
        csv_file.flush()
        history.append(row)

        # -- Console log every 10 episodes ----------------------------------
        if ep % 10 == 0 or ep == 1:
            print(
                f"Ep {ep:4d}/{episodes} | "
                f"R={ep_reward:7.2f} | "
                f"steps={ep_steps:4d} | "
                f"faults={fault_steps:3d} ({fault_rate:.0%}) | "
                f"c_loss={avg_critic:.4f} | "
                f"a_loss={avg_actor:.4f} | "
                f"α={avg_alpha:.3f} | "
                f"{elapsed:.0f}s"
            )

        # -- Checkpoint ------------------------------------------------------
        if ep % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"sac_ep{ep:04d}.pt")
            agent.save(ckpt_path)
            print(f"  [checkpoint] Saved → {ckpt_path}")

    # -- Save final checkpoint -----------------------------------------------
    final_path = os.path.join(save_dir, "sac_final.pt")
    agent.save(final_path)
    print(f"\nTraining complete. Final model → {final_path}")
    print(f"Training log    → {log_path}")

    csv_file.close()
    env.close()

    # -- Evaluation ----------------------------------------------------------
    if eval_episodes > 0:
        _evaluate(agent, fault_prob=fault_prob, episodes=eval_episodes)

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    agent: SACAgent,
    fault_prob: float = 0.15,
    episodes: int = 5,
    max_steps: int = 500,
) -> None:
    """Run greedy evaluation episodes and print a summary table."""
    print(f"\n{'=' * 55}")
    print(f"EVALUATION ({episodes} episodes, fault_prob={fault_prob})")
    print(f"{'=' * 55}")

    env = UR3SimEnv(render_mode=None, fault_prob=fault_prob, max_steps=max_steps)
    rewards, fault_rates = [], []

    for ep in range(1, episodes + 1):
        obs = _unwrap_reset(env.reset())
        ep_reward, fault_steps, ep_steps = 0.0, 0, 0
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = _unwrap_step(env.step(action))
            ep_reward += reward
            ep_steps += 1
            if info.get("fault", "none") != "none":
                fault_steps += 1
        fr = fault_steps / max(ep_steps, 1)
        rewards.append(ep_reward)
        fault_rates.append(fr)
        print(f"  Eval ep {ep:3d} | R={ep_reward:7.2f} | fault_rate={fr:.0%}")

    print(f"{'─' * 45}")
    print(f"  Mean reward    : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean fault rate: {np.mean(fault_rates):.1%}")
    print(f"{'=' * 55}")
    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SAC agent on LEASAR UR3 simulation."
    )
    p.add_argument("--episodes",      type=int,   default=500)
    p.add_argument("--max_steps",     type=int,   default=500)
    p.add_argument("--fault_prob",    type=float, default=0.15)
    p.add_argument("--warmup_steps",  type=int,   default=1000)
    p.add_argument("--save_dir",      type=str,   default="checkpoints")
    p.add_argument("--save_every",    type=int,   default=50)
    p.add_argument("--log_path",      type=str,   default="results/sac_training_log.csv")
    p.add_argument("--eval_episodes", type=int,   default=5)
    p.add_argument("--render",        action="store_true")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        fault_prob=args.fault_prob,
        warmup_steps=args.warmup_steps,
        save_dir=args.save_dir,
        save_every=args.save_every,
        log_path=args.log_path,
        eval_episodes=args.eval_episodes,
        render=args.render,
        seed=args.seed,
    )
