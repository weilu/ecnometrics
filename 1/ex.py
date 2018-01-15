import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy import stats
import math


def read_data(infile):
    with open(infile) as f:
        f.readline()  # skip the header
        return np.loadtxt(f)


def ex1():
    data = read_data("TrainExer11.txt")
    ages = data[:, 1]
    expenditures = data[:, 2]

    # plot hist
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(ages)
    axs[1].hist(expenditures)
    axs[0].set_xlabel("Age")
    axs[1].set_xlabel("Spend")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Age & Spend Histogram")

    # scatter plot
    f, ax = plt.subplots()
    ax.scatter(ages, expenditures)
    ax.set_title('Age vs Spend')

    plt.show()


def ex3():
    data = read_data("TrainExer13.txt")
    games = data[:, 0]
    time = data[:, 2]

    params = stats.linregress(games, time)
    r2 = params.rvalue ** 2
    ystd = np.std(np.array(time), ddof=2)
    s = math.sqrt((ystd ** 2) * (1 - r2))
    a = params.intercept
    b = params.slope
    print(f'a: {round(a, 3)}, b: {round(b, 3)}, r2: {round(r2, 3)}, s: {round(s, 3)}')
    print(f'About {round(r2 * 100)}% of the variance in winning time can be explained by the game number, so it is a reasonably good estimator')
    last_game = int(max(time))
    predictions = list(map(lambda x: round(b * x + a, 2), range(last_game + 1, last_game + 4)))
    print(f'predictions for 2008, 2012, 2016 are: {predictions}, vs actual: 9.96, 9.63, 9.81')

    z = np.polyfit(games, time, 1)
    p = np.poly1d(z)
    f, ax = plt.subplots()
    xp = np.linspace(min(games), max(games), 100)
    ax.plot(games, time, '.', xp, p(xp), '-')
    ax.set_title('Game vs winning time (men)')
    ax.set_xlabel("Game #")
    ax.set_ylabel("Time")
    plt.show()


if __name__ == '__main__':
    file_path = os.path.dirname(__file__)
    os.chdir(file_path)
    # ex1()
    ex3()

