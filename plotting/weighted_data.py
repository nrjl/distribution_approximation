import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

def make_rgb_array(n_colours, colours=None):
        # This will either use the system default colours, or specified list (and will just loop over if there aren't enough colours)
        if colours is None:
            colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        return np.array([to_rgb(colours[i%len(colours)]) for i in range(n_colours)])

class WeightedData:
    data=None
    weights=None
    _colour_array=None

    def __init__(self, ax, data, weights=None, colours=None, **kwargs) -> None:

        self.set_data(data)
        self.set_weights(weights) 

        # This will either use the system default colours, or specified list (and will just loop over if there aren't enough colours)
        self.rgb_colours = make_rgb_array(self.weights.shape[1], colours)
        self._update_colour_array()

        self._artists = [ax.scatter(self.data[:,0], self.data[:,1], c=self._colour_array, **kwargs)]

    def get_artists(self):
        return self._artists

    def set_data(self, data):
        data = np.array(data)
        if data.ndim == 1:
            data = np.hstack((np.reshape(data, (len(data), 1)), np.zeros((len(data), 1))))
        elif data.ndim > 2 or data.shape[1] > 2:
            raise ValueError('Input data as a 1D or 2D n*2 array')
        self.data = data
        
    def set_weights(self, weights=None):

        if weights is None:
           weights = np.ones((self.data.shape[0], 1))
    
        assert weights.shape[0] == self.data.shape[0], "Weights must have same first dimension as data"

        # Normalise
        self.weights = weights/weights.sum(axis=1, keepdims=True)

    def _update_colour_array(self):
        self._colour_array = np.matmul(self.weights, self.rgb_colours)

    def update(self, data=None, weights=None):
        if data is not None:
            self.set_data(data)
            self._artists[0].set_offsets(self.data)
        if weights is not None:
            self.set_weights(weights)
            self._update_colour_array()
            self._artists[0].set_color(self._colour_array)

        return self._artists





        



        

