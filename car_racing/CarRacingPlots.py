import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_performance(scores, title = 'Title', window = 50):

    plt.figure(figsize=(12,7))
    sns.set_context("talk")

    df = pd.DataFrame(scores, columns=["Reward"])
    temperature = df['Reward']
    t_average = temperature.rolling(window=window).mean()

    plt.plot(temperature, 'k-', label='Original', alpha=0.25)
    plt.plot(t_average, '#33cccc', label='Running average - 50 episodes')
    plt.ylabel('Sum of Rewards')
    plt.xlabel('Episode')
    plt.grid(linestyle=':')
    plt.fill_between(t_average.index, 0, t_average, color='#33cccc', alpha=0.05)
    plt.legend(loc='upper left')
    plt.title(title, fontsize=30 )
    plt.show();

def plot_episode_duraton(episode_durations, title = 'Episode Duration'):

    plt.figure(figsize=(12,7))
    sns.set_context("talk")

    plt.plot(episode_durations, '#33cccc', label='Episode Duration')
    plt.ylabel('# Time Steps')
    plt.xlabel('Episode')
    plt.grid(linestyle=':')
    plt.legend(loc='upper left')
    plt.title(title, fontsize=30 )
    plt.show();


def plot_epsilon(epsilons, title = 'Epsilon Values'):

    plt.figure(figsize=(12,7))
    sns.set_context("talk")

    plt.plot(epsilons, '#33cccc', label='ε - value')
    plt.ylabel('Epsilon (ε)')
    plt.xlabel('Episode')
    plt.grid(linestyle=':')
    plt.legend(loc='upper right')
    plt.title(title, fontsize=30 )
    plt.show();
