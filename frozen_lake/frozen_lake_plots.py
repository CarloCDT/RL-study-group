import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np

def plot_frozen_lake(df_summary, terminal_states = [5, 7, 11, 12, 15], size = (4,4)):

    # Label dict
    direction_dict = {0:"←", 1:"↓", 2:"→", 3:"↑"}
    
    df_temp = df_summary.copy()
    df_temp["label"] = df_temp["optimal_action"].apply(lambda x: direction_dict[x])
    df_temp["x"] = df_temp["state"].apply(lambda x: x//size[0])
    df_temp["y"] = df_temp["state"].apply(lambda x: x%size[1])

    df_temp = df_temp[~df_temp["state"].isin(terminal_states)]

    # Reshape Data
    returns = df_temp.pivot("x", "y", "state_value")
    direction = df_temp.pivot("x", "y", "label")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16,7))
    fig.suptitle('State Values for Optimal Policy')

    sns.heatmap(returns, annot=True, linewidths=.5, cbar=False, cmap="Blues", fmt = '.2f', ax=axes[0])
    sns.heatmap(returns, annot=np.array(direction), linewidths=.5, cbar=False, cmap="Blues", fmt = '', ax=axes[1])

def plot_state_action_frozen_lake(df_summary, terminal_states = [5, 7, 11, 12, 15], size = (4,4)):

    direction_dict = {0:"←", 1:"↓", 2:"→", 3:"↑"}

    df_temp = df_summary.copy()
    df_temp["label"] = df_temp["optimal_action"].apply(lambda x: direction_dict[x])
    df_temp["x"] = df_temp["state"].apply(lambda x: x//size[0])
    df_temp["y"] = df_temp["state"].apply(lambda x: x%size[1])

    df_temp = df_temp[~df_temp["state"].isin(terminal_states)]
    df_temp = df_temp[["x","y",0,1,2,3]]

    df_piv = df_temp.pivot_table(index='x', columns='y')

    M = len(df_piv.columns) // 4
    N = len(df_piv)
    values = [df_piv[dir] for dir in
            [3, 2, 1, 0]]  # {0: '←', 1: '↓', 2: '→', 3: '↑'}

    # Triangulation
    xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
    xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    cstart = (M + 1) * (N + 1)  # indices of the centers

    trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    triangul = [Triangulation(x, y, triangles) for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]


    # Plot
    cmaps = ['Blues'] * 4

    norms = [plt.Normalize(0, 1) for _ in range(4)]
    fig, ax = plt.subplots(figsize=(16, 7))
    imgs = [ax.tripcolor(t, np.ravel(val), cmap=cmap, norm=norm, ec='white')
            for t, val, cmap, norm in zip(triangul, values, cmaps, norms)]


    for val, dir in zip(values, [(-1, 0), (0, 1), (1, 0), (0, -1)]):
        for i in range(M):
            for j in range(N):
                v = val[i][j]
                ax.text(i + 0.3 * dir[1], j + 0.3 * dir[0], f'{v:.2f}', color='k' if 0.2 < v < 0.8 else 'w', ha='center', va='center')

    ax.tick_params(length=0)
    ax.set_xticks(range(M))
    ax.set_xticklabels(df_piv[3].columns)
    ax.set_yticks(range(N))
    ax.set_yticklabels(df_piv.index)
    ax.invert_yaxis()
    ax.margins(x=0, y=0)

    ax.set_title("State-Action Values")

    plt.tight_layout()
    plt.show()
