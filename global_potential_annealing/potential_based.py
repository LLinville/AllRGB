import numpy as np

all_intensities = np.linspace(0,1,256)
r,g,b = np.array(np.meshgrid(all_intensities, all_intensities, all_intensities)).T.reshape(-1,3)
