# Cartpole 泛化能力探究 / Exploring Generalization in Cartpole

## 观察与问题 / Observation & Question

在设定 Cartpole 的胜利步数为 200 步时，我观察到，这个模型只能存活到 230 步左右。虽然能完成任务，但是足以证明模型只学会了在 200 步以内存活，却没有泛化能力。

When the victory threshold for Cartpole is set to 200 steps, I observed that the model can only survive for around 230 steps. Although it is able to complete the task, this sufficiently demonstrates that the model has only learned to survive within 200 steps and lacks generalization capability.

若将胜利步数设置为 500 或者更多时，模型能不能有泛化能力？这是我希望探究的一个问题。

If the victory threshold is set to 500 steps or more, will the model be able to generalize? This is a question I wish to explore.