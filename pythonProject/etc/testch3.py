import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

anscombe_data = np.load("../data/ch3_anscombe.npy")

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

c = ax1.hist2d( anscombe_data[0,:, 0],
              anscombe_data[0,:, 1],
              bins=[8, 5], range=[(4,20),(3,13)])
c = ax2.hist2d( anscombe_data[1,:, 0],
              anscombe_data[1,:, 1],
              bins=[8, 5], range=[ (4,20),(3,13)])

c = ax3.hist2d( anscombe_data[2,:, 0],
              anscombe_data[2,:, 1],
              bins=[8, 5], range=[(4,20),(3,13)])

c = ax4.hist2d( anscombe_data[3,:, 0],
              anscombe_data[3, :, 1],
              bins=[8, 5], range=[(4,20),(3, 13)])

ax1.set_xticks(c[1])
ax1.set_yticks(c[2])
plt.show()