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
    plt.show()
