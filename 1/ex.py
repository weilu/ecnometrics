import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os


def ex1():
    file_path = os.path.dirname(__file__)
    os.chdir(file_path)
    with open("TrainExer11.txt") as f:
        f.readline()  # skip the header
        data = np.loadtxt(f)

        # plot hist
        ages = data[:, 1]
        expenditures = data[:, 2]
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


if __name__ == '__main__':
    ex1()

