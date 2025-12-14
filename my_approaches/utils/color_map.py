import matplotlib.pyplot as plt


def get_colors_for_trees(number_of_trees):
    return plt.cm.viridis([i / number_of_trees for i in range(number_of_trees)])

def get_color_for_metrics(number_of_metrics):
    # colors = plt.cm.plasma([i / number_of_metrics for i in range(number_of_metrics)])
    # colors = plt.cm.hsv([i / number_of_metrics for i in range(number_of_metrics)])
    # colors = plt.cm.rainbow([i / number_of_metrics for i in range(number_of_metrics)])
    colors = plt.cm.ocean([i / number_of_metrics for i in range(number_of_metrics)])
    return colors
