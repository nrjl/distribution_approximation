import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.collections import PatchCollection
import matplotlib.animation as anim

fig, ax = plt.subplots()
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])

ellipses = []
n_ellipses = 3

for i in range(n_ellipses):
    ellipse = Ellipse((i, 0), width=2, height=2, animated=True, alpha=0.5)
    ellipses.append(ellipse)


n_frames = 200

def init():
    for e in ellipses:
        ax.add_patch(e)
    return ellipses

def animate(i): 
    for j in range(3):
        #tf = transforms.Affine2D().translate(0, j*(1.0 - i/n_frames)
                #.scale(j/2.0*i/n_frames, 1) \
                #.rotate(i/n_frames*2*np.pi + j*np.pi/2) \
                #.translate(0, j*i/n_frames)
        ellipses[j].set_width(i/n_frames)
        ellipses[j].set_height(2*i/n_frames)
        ellipses[j].set(transform=transforms.Affine2D().rotate_deg(j*360.0*i/n_frames))

    return ellipses

a = anim.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=10, blit=True) 
plt.show()