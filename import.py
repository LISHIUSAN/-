# Q_learning最佳路径探索设计
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# 设置网格大小
GRID_SIZE = 10
# 设置起始状态和目标状态
START_STATE = (0, 0)
GOAL_STATE = (GRID_SIZE-1, GRID_SIZE-1)
# 设置 Q-Learning 算法的参数
EPSILON = 0.1  # 探索概率
ALPHA = 0.5    # 学习率
GAMMA = 0.99   # 折扣因子

# 初始化 Q 表,Q 表记录每个状态下选择每个动作的预期回报
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))

def q_learning(start_state, goal_state):
    # 初始化状态和步数
    state = start_state
    step = 0
    path = [start_state]  # 记录路径

    # 循环直到达到目标状态
    while state != goal_state:
        # 选择动作
        # 如果随机数小于探索概率,选择随机动作
        if np.random.rand() < EPSILON:
            action = np.random.randint(4)
        # 否则选择 Q 表中当前状态下回报最大的动作
        else:
            action = np.argmax(Q_table[state])

        # 执行动作并观察奖励和下一状态
        if action == 0:
            next_state = (state[0]-1, state[1])
        elif action == 1:
            next_state = (state[0]+1, state[1])
        elif action == 2:
            next_state = (state[0], state[1]-1)
        else:
            next_state = (state[0], state[1]+1)

        # 确保下一状态在网格内
        next_state = (max(0, min(next_state[0], GRID_SIZE-1)),
                      max(0, min(next_state[1], GRID_SIZE-1)))

        # 计算奖励
        if next_state == goal_state:
            reward = 100
        else:
            reward = -1

        # 更新 Q 表
        Q_table[state][action] = (1 - ALPHA) * Q_table[state][action] + \
                                ALPHA * (reward + GAMMA * np.max(Q_table[next_state]))

        # 更新状态和路径
        state = next_state
        path.append(state)
        step += 1

    return step, path
def plot_grid(path, goal_state):
    """
    绘制 Q-Learning 算法找到的路径。

    参数:
    path (list): 路径坐标列表,每个元素是一个(x, y)元组。
    goal_state (tuple): 目标状态的坐标(x, y)。

    返回:
    无
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(np.arange(0, GRID_SIZE, 1))
    ax.set_yticks(np.arange(0, GRID_SIZE, 1))
    ax.grid()

    # 绘制网格
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) == goal_state:
                # 将目标状态绘制为绿色
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='green', edgecolor='black'))
            else:
                # 其他格子绘制为白色
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='black'))

    # 绘制路径
    path_x = [p[1] for p in path]
    path_y = [p[0] for p in path]
    ax.plot(path_x, path_y, color='red', linewidth=2)

    plt.show()

def plot_grid_animation(path, goal_state):
    """
    绘制 Q-Learning 算法找到的路径动画。

    参数:
    path (list): 路径坐标列表,每个元素是一个(x, y)元组。
    goal_state (tuple): 目标状态的坐标(x, y)。

    返回:
    无
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(np.arange(0, GRID_SIZE, 1))
    ax.set_yticks(np.arange(0, GRID_SIZE, 1))
    ax.grid()

    # 绘制网格
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) == goal_state:
                # 将目标状态绘制为绿色
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='green', edgecolor='black'))
            else:
                # 其他格子绘制为白色
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='black'))

    # 绘制路径
    path_x = [p[1] for p in path]
    path_y = [p[0] for p in path]
    line, = ax.plot([], [], color='red', linewidth=2)

    def animate(frame):
        # 在每一帧中更新路径的绘制
        line.set_data(path_x[:frame+1], path_y[:frame+1])
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=len(path), interval=500, repeat=False)
    plt.show()

# 运行 Q-learning 算法并可视化探索过程
steps, path = q_learning(START_STATE, GOAL_STATE)

# 绘制静态图像
plot_grid(path, GOAL_STATE)

# 绘制动态图像
plot_grid_animation(path, GOAL_STATE)

# 运行 Q-learning 算法
steps = []
for i in range(1000):
    steps.append(q_learning(START_STATE, GOAL_STATE)[0])

# 绘制收敛曲线
plt.figure(figsize=(8, 6))
plt.plot(steps)
plt.title("Q-learning Convergence")
plt.xlabel("Episode")
plt.ylabel("Steps to Goal")
plt.show()