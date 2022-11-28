import numpy as np
import matplotlib.pyplot as plt
from mixture_models import MixtureModel, GaussianComponent
import matplotlib.animation as anim
import argparse

parser = argparse.ArgumentParser(description='Fit and plot a GMM using Expectation Maximisation')
parser.add_argument('-c', '--components', type=int, default=3, help='Number of Gaussian components')
parser.add_argument('-d', '--dimensions', type=int, default=2, help='Number of dimensions (only 1 or 2 supported)')
parser.add_argument('-o', '--observations', type=int, default=2000, help='Number of observations')
parser.add_argument('-s', '--steps', type=int, default=100, help='Number of EM steps')
parser.add_argument('-a', '--save-animation', action='store_true', help='Save animation')
args = parser.parse_args()

true_gmm = MixtureModel(GaussianComponent, args.components, args.dimensions)
fit_gmm = MixtureModel(GaussianComponent, args.components, args.dimensions)

Z = true_gmm.sample(args.observations)
fit_gmm.add_observations(Z)

fig, ax  = plt.subplots()

def axlim(low, high, pad=0.1):
    low = low - pad*(high-low)
    high = high + pad*(high-low)
    return low, high

n_steps = 100

def init():
    ax.cla()
    
    ax.set_xlim(*axlim(Z[:,0].min(), Z[:,0].max()))
    ax.set_ylim(*axlim(Z[:,1].min(), Z[:,1].max()))

    h_true = true_gmm.plot_init(ax, color='k', animated=False, facecolor='none')

    fit_gmm._init_mixture_params()
    h_fit = fit_gmm.plot_init(ax, plot_observations=True)

    return h_fit

def animate(i):
    delta = fit_gmm.EM_step()
    print('Step {0:03d}, max parameter change: {1}'.format(i, delta))
    h_fit = fit_gmm.plot_update()
    return h_fit

a = anim.FuncAnimation(fig, animate, init_func=init, frames=n_steps, interval=100, blit=True)
if args.save_animation:
    a.save('movies/gmm_em_fit.mp4', writer='ffmpeg', dpi=200,
        extra_args=["-crf", "18", "-profile:v", "main", "-tune", "animation", "-pix_fmt", "yuv420p"])

plt.show()