# %%
%matplotlib ipympl
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %%
stepsize = 0.5
num_steps = 20
num_trials = 5

final_position = []

for _ in range(num_trials):
    pos = np.array([0, 0])
    path = []
    for i in range(num_steps):
        pos = pos + np.random.normal(0, stepsize, 2)
        path.append(pos)
    final_position.append(np.array(path))
    
x = [final_position[i][:,0] for i in range(len(final_position))]
y = [final_position[j][:,1] for j in range(len(final_position))]

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot()
fig.subplots_adjust(left=0.1, right=0.85)

cmap = plt.get_cmap('tab10')

def animate(frame):
    step_num = frame % (num_steps)
    trial_num = frame//(num_steps)
    color = cmap(trial_num % 10)
    if step_num == num_steps-1:
        label = f"Trial = {trial_num+1}"
    else:
        label = None
    ax.plot(x[trial_num][:step_num], y[trial_num][:step_num], color = color, ls = '-',linewidth = 0.5,
            marker = 'o', ms = 8, mfc = color, mec ='k', zorder = trial_num, label = label)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"Number of trials = {trial_num+1} \nNumber of steps = {step_num+1}")  
    if step_num == num_steps-1:
        ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)
    
    return ax

fig.suptitle(f"2D random walk simulation for {num_steps} steps over {num_trials} trials.")
ani = FuncAnimation(fig, animate, frames= np.arange(0, (num_steps * num_trials)), interval = 100, repeat = False)
ani;

# %%
stepsize = 0.5
num_steps = 20
num_trials = 5

final_position = []

for _ in range(num_trials):
    pos = np.array([0, 0])
    path = []
    for i in range(num_steps):
        pos = pos + np.random.normal(0, stepsize, 2)
        path.append(pos)
    final_position.append(np.array(path))
    
x = [final_position[i][:,0] for i in range(len(final_position))]
y = [final_position[j][:,1] for j in range(len(final_position))]

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot()
fig.subplots_adjust(left=0.1, right=0.85)

cmap = plt.get_cmap('tab10')

def animate(frame):
    step_num = frame % (num_steps)
    trial_num = frame//(num_steps)
    color = cmap(trial_num % 10)
    if step_num == num_steps-1:
        label = f"Trial = {trial_num+1}"
    else:
        label = None
    ax.plot(x[trial_num][:step_num], y[trial_num][:step_num], color = color, ls = '-',linewidth = 0.5,
            marker = 'o', ms = 8, mfc = color, mec ='k', zorder = trial_num, label = label)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"Number of trials = {trial_num+1} \nNumber of steps = {step_num+1}")  
    if step_num == num_steps-1:
        ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)
    
    return ax

fig.suptitle(f"2D random walk simulation for {num_steps} steps over {num_trials} trials.")
ani = FuncAnimation(fig, animate, frames= np.arange(0, (num_steps * num_trials)), interval = 100, repeat = False)
ani;
