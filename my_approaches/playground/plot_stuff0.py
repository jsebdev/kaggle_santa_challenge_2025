# %%
%matplotlib ipympl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))

# Create a text box inside the plot area
text = ax.text(
    0, 0,
    # 0.05, 0.95,                # x, y in axis coordinates (0–1)
    "",                        # initial text
    transform=ax.transAxes,    # use axes-relative positioning
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle="round", fc="white", ec="black")
)

def update(frame):
    # Update line
    line.set_ydata(np.sin(x + frame/10))

    # Update the text box
    text.set_text(f"Frame: {frame}")

    return line, text

ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# plt.show()
from IPython.display import HTML
plt.close()
HTML(ani.to_jshtml())  # Display animation as HTML5 video
