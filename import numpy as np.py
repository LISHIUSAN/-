#设计最优调节对比主流算法（SARSA）
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# 参数设置
GRID_SIZE = 10  # 网格世界的大小
START_STATE = (0, 0)  # 起始状态的坐标
GOAL_STATE = (GRID_SIZE-1, GRID_SIZE-1)  # 目标状态的坐标
EPSILON = 0.1  # 探索概率
ALPHA = 0.5  # 学习率
GAMMA = 0.99  # 折扣因子
# 初始化 Q 表
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
def sarsa(start_state, goal_state):
    """
    SARSA 算法实现
    """
    EPSILON1 = 0.5
    ALPHA1 = 0.5
    GAMMA1 =0.9
    current_state = start_state
    path = [current_state]
    steps = 0

    while current_state != goal_state:
        # 根据 epsilon-greedy 策略选择动作
        if random.random() < EPSILON1:
            # 探索: 随机选择一个动作
            action = random.randint(0, 3)
        else:
            # 利用: 选择 Q 值最大的动作
            action = np.argmax(Q_table[current_state[0], current_state[1]])

        # 执行动作并获得下一状态和奖励
        next_state = get_next_state(current_state, action)
        reward = get_reward(current_state, next_state, goal_state)

        # 更新 Q 表
        next_action = np.argmax(Q_table[next_state[0], next_state[1]])
        Q_table[current_state[0], current_state[1], action] += ALPHA1 * (reward + GAMMA1 * Q_table[next_state[0], next_state[1], next_action] - Q_table[current_state[0], current_state[1], action])

        # 更新状态和路径
        current_state = next_state
        path.append(current_state)
        steps += 1

    return steps, path

def q_learning(start_state, goal_state):
    """
    Q-Learning 算法实现
    """
    current_state = start_state
    path = [current_state]
    steps = 0

    while current_state != goal_state:
        # 根据 epsilon-greedy 策略选择动作
        if random.random() < EPSILON:
            # 探索: 随机选择一个动作
            action = random.randint(0, 3)
        else:
            # 利用: 选择 Q 值最大的动作
            action = np.argmax(Q_table[current_state[0], current_state[1]])

        # 执行动作并获得下一状态和奖励
        next_state = get_next_state(current_state, action)
        reward = get_reward(current_state, next_state, goal_state)

        # 更新 Q 表
        Q_table[current_state[0], current_state[1], action] += ALPHA * (reward + GAMMA * np.max(Q_table[next_state[0], next_state[1]]) - Q_table[current_state[0], current_state[1], action])

        # 更新状态和路径
        current_state = next_state
        path.append(current_state)
        steps += 1

    return steps, path

def get_next_state(state, action):
    """
    根据当前状态和动作获得下一状态
    """
    x, y = state
    if action == 0:
        return (x, min(y + 1, GRID_SIZE - 1))
    elif action == 1:
        return (min(x + 1, GRID_SIZE - 1), y)
    elif action == 2:
        return (x, max(y - 1, 0))
    else:
        return (max(x - 1, 0), y)

def get_reward(current_state, next_state, goal_state):
    """
    根据当前状态和下一状态计算奖励
    """
    if next_state == goal_state:
        return 1.0
    else:
        return -0.1

# 运行 SARSA 和 Q-Learning 算法多次并绘制收敛曲线
q_learning_steps = []
sarsa_steps = []

for i in range(1000):
    q_learning_steps.append(q_learning(START_STATE, GOAL_STATE)[0])
    sarsa_steps.append(sarsa(START_STATE, GOAL_STATE)[0])

plt.figure(figsize=(8, 6))
plt.plot(q_learning_steps, label='Q-Learning')
plt.plot(sarsa_steps, label='SARSA')
plt.title("Convergence Comparison")
plt.xlabel("Episode")
plt.ylabel("Steps to Goal")
plt.legend()
plt.show()