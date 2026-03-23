# 冰湖环境强化学习实验说明  
# Reinforcement Learning Experiment on FrozenLake

## 📁 文件说明 | File Description

- **`train_1.py`**  
  实现了标准的 **Q 学习（Q-Learning）** 算法，应用于 **非湿润（确定性）** 的冰湖环境（`FrozenLake-v1`，参数 `is_slippery=False`）。  
  在该设定下，环境转移是确定性的，智能体可以精确学习每一步的最优动作。

- **`train_1.py`**  
  Implements standard **Q-Learning** on a **non-slippery (deterministic)** FrozenLake environment (`FrozenLake-v1` with `is_slippery=False`).  
  Under this setting, state transitions are deterministic, allowing the agent to learn the optimal action for each step precisely.

---

## 📊 实验结果 | Experimental Results

- **非湿润（确定性）环境**  
  经过约 **10,000 步** 的训练，智能体在测试中获得了 **0.88** 的获胜概率（即 88% 的成功率），表现出良好的收敛效果。

- **Non‑slippery (deterministic) environment**  
  After approximately **10,000 training steps**, the agent achieves a **win rate of 0.88** (88% success) during evaluation, demonstrating good convergence.

- **湿润（随机）环境**  
  然而，将同样的 Q 学习算法直接应用于 **湿润（`is_slippery=True`）** 环境时，效果明显下降。由于动作执行存在随机偏移（滑冰效果），确定性策略难以有效应对随机性，导致成功率远低于非湿润情形。

- **Slippery (stochastic) environment**  
  However, applying the same Q‑Learning algorithm directly to the **slippery (`is_slippery=True`)** environment yields significantly worse performance. The stochastic transitions (slip effects) make the deterministic policy ineffective, resulting in a much lower success rate compared to the non‑slippery case.

---

## 🚀 下一步计划 | Next Steps

为了解决湿润环境下的随机性问题，我计划 **新建一个独立的文件**，专门探究适用于随机环境的强化学习方法。可能的改进方向包括：

- **期望 SARSA（Expected SARSA）**：通过期望值更新降低方差，提高对随机性的鲁棒性。
- **使用深度 Q 网络（DQN）**：利用神经网络近似 Q 值，更好地处理环境不确定性。

新文件将对比不同算法在湿润环境下的表现，并尝试找到能够稳定达到高胜率的解决方案。

To address the stochasticity in the slippery environment, I plan to **create a separate file** dedicated to exploring reinforcement learning methods suitable for stochastic environments. Potential improvements include:

- **Expected SARSA**: Reducing variance via expectation updates, which is more robust to randomness.
- **Deep Q‑Network (DQN)**: Using neural networks to approximate Q‑values, better handling environmental uncertainty.

The new file will compare the performance of different algorithms in the slippery setting and aim to find a solution that consistently achieves a high win rate.
