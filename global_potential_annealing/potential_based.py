import numpy as np
from matplotlib import pyplot as plt

def potential(grid):
    

color_depth = 16
all_intensities = np.linspace(0, 1, color_depth)
all_colors = np.array(np.meshgrid(all_intensities, all_intensities, all_intensities)).T.reshape(-1,3)
# np.random.shuffle(all_colors)
grid = all_colors.reshape(-1,np.int(color_depth*np.sqrt(color_depth)),3)
print(grid[0,0])
plt.imshow(grid)
plt.show()
plt.pause(0.001)

