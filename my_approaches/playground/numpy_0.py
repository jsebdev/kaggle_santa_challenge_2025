# %%
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

frame = 5

xp = x[:frame]
yp = y[:frame]
print('>>>>> numpy_0.py:11 "np.stack([xp, yp])"')
print(np.stack([xp, yp]))
print('>>>>> numpy_0.py:13 "np.stack([xp, yp]).T"')
print(np.stack([xp, yp]).T)
print('>>>>> numpy_0.py:15 "np.stack([xp, yp], axis=1)"')
print(np.stack([xp, yp], axis=1))
