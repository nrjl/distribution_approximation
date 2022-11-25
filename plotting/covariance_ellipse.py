import numpy as np
from matplotlib.patches import Ellipse


class CovarianceEllipses:
     
    def __init__(self, ax, mean=[0,0], cov=[[1,0],[0,1]], ellipse_masses=[0.68, 0.95], alpha=0.3, animated=True, facecolor=None, **kwargs) -> None:
        self.ax = ax
        self._ellipse_scales = np.sqrt(-2*np.log(1-np.array(ellipse_masses)))
        self._artists = []
        d1, d2, angle = self.get_ellipse_params(cov)

        for s in self._ellipse_scales:
            ellipse = Ellipse(mean, width=d1*s, height=d2*s, angle=angle, alpha=alpha, animated=animated, **kwargs)
            if facecolor is 'none':
                ellipse.set_facecolor('none')
            ax.add_patch(ellipse)
            self._artists.append(ellipse)

    def get_artists(self):
        return self._artists

    def update(self, mean, cov):
        d1, d2, angle = self.get_ellipse_params(cov)
        for ell, s in zip(self._artists, self._ellipse_scales):
              ell.set_width(d1*s)
              ell.set_height(d2*s)
              ell.set_angle(angle)
              ell.set_center(mean)

        return self._artists
        
    @staticmethod     
    def get_ellipse_params(cov):
        # Get ellipsae parameters (note angle returned in degrees because that's what Ellipse.set_angle() expects)
        # Get eigenvalues and sort them (big first)
        w, V = np.linalg.eig(cov)
        idx = -w.argsort()[::-1]   
        w = w[idx]
        V = V[:,idx]
        d1, d2 = 2*np.sqrt(w)
        angle = np.arctan2(V[0,1], V[0,0])*180.0/np.pi
        return d1, d2, angle