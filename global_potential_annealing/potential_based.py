import numpy as np
from matplotlib import pyplot as plt
from itertools import count, product
from random import randint

def ColorDistance(rgb1,rgb2):
    '''d = {} distance between two colors(3)'''
    rm = 0.5*(rgb1[...,0]+rgb2[...,0])
    d = np.sum(np.sum((2+rm,4,3-rm)*(rgb1-rgb2)**2, axis=2)[0,0]**0.5)
    return d

def perceptual_smoothness(grid):
    return ColorDistance(grid, np.roll(grid, [-1,-1])) + \
           ColorDistance(grid, np.roll(grid, [1,-1])) + \
           ColorDistance(grid, np.roll(grid, [-1,1])) + \
           ColorDistance(grid, np.roll(grid, [1,1]))

def potential(grid):
    width = grid.shape[0]
    total = np.zeros_like(grid)

    sharpness = 2
    x, y = np.meshgrid(np.linspace(-1*sharpness, sharpness, width * 2 - 1), np.linspace(-1*sharpness, sharpness, width * 2 - 1))
    dropoff = 1 / (x*x*x*x + y*y*y*y + 0.1)
    dropoff = dropoff[..., np.newaxis]
    for x in range(0, grid.shape[0], 1):
        # print(x)
        for y in range(0, grid.shape[1], 1):
            total += grid[x,y] * dropoff[width-1-x:2*width-1-x, width-1-y:2*width-1-y]

    return -1 * total / (width*width)



color_depth = 4
all_intensities = np.linspace(0, 1, color_depth)
all_colors = np.array(np.meshgrid(all_intensities, all_intensities, all_intensities)).T.reshape(-1,3)
np.random.shuffle(all_colors)
grid = all_colors.reshape(-1,np.int(color_depth*np.sqrt(color_depth)),3)
width = grid.shape[0]
print(grid[0,0])
# plt.imshow(grid)
# plt.show()
# plt.pause(0.001)

pot = potential(grid)

for i in count():
    x1, y1 = randint(0, width - 1), randint(0, width - 1)
    # for dx,dy in product(range(-10,10),range(-10,10)):
    dx, dy = randint(-5, 5), randint(-5, 5)
    x2, y2 = x1 + dx, y1 + dy
    if i%1 == 0:
        x2, y2 = randint(0, width - 1), randint(0, width - 1)

    if not 0 <= x2 <= width-1 or not 0 <= y2 <= width-1:
        continue
    p1, p2 = grid[x1,y1], grid[x2,y2]
    pot_change = (p2 * pot[x1,y1] + p1 * pot[x2,y2]) - (p1 * pot[x1,y1] + p2 * pot[x2,y2])
    if np.sum(pot_change) < 0:
        temp = p1.copy()
        grid[x1,y1] = p2
        grid[x2,y2] = temp


    if i%1000 == 0:
        print(i)
    # if i%100000 == 0:
    #     print(f"Smoothness: {perceptual_smoothness(grid)}")
    if i%100 == 0:
        pot = potential(grid)
    if i%100 == 0:
        plt.imshow(grid)
        plt.pause(0.001)