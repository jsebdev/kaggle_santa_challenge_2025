import matplotlib.pyplot as plt


def get_colors(number_of_trees):
    return plt.cm.viridis([i / number_of_trees for i in range(number_of_trees)])

