import numpy as np
from scipy.stats import wishart, multivariate_normal
from plotting.covariance_ellipse import CovarianceEllipses
from plotting.weighted_data import WeightedData, make_rgb_array

class MixtureModel:
    weight = None

    def __init__(self, component_model, n_components: int, problem_dimension: int=2, observations=None) -> None:
        # Mixture Model
        # Component type
        # n_components (commonly K)
        # observations Z

        assert issubclass(component_model, Component), "Component model must be Component subclass"
        self.component_model = component_model
        self.n_components = n_components

        assert problem_dimension > 0, "Problem dimension must be int > 0"
        self.d = problem_dimension
        self.observations = np.empty([1, self.d])
        if observations is not None:
            self.add_observations(observations)
        self._init_mixture_params()

    def sample(self, n: int=1):
        # Generate a set of samples from the current mixture model
        Z = np.zeros((n, self.d))

        # Determine how many samples to draw from each component (sample multinomial with component weights)
        n_samples = np.random.multinomial(n, self.component_weights)
        zc = 0

        # Generate samples from each component
        for i in range(self.n_components):
            Z[zc:zc+n_samples[i],:] = self.components[i].sample(n_samples[i])
            zc += n_samples[i]
        return Z

    def add_observations(self, Z):
        Z = np.atleast_2d(Z)
        assert Z.shape[1] == self.d, "Number of dimensions must match existing observations, new = {0}, existing = {1}".format(Z.shape[1], self.d)
        self.observations = np.vstack((self.observations, Z))
         
    def _init_mixture_params(self):
        self.component_weights = np.random.dirichlet(np.ones(self.n_components))     # AKA Phi
        self.components = [self.component_model() for i in range(self.n_components)]

    def data_likelihood(self):
        p_z = np.zeros((self.observations.shape[0], self.n_components))
        for i in range(self.n_components):
            p_z[:, i] = self.components[i].likelihood(self.observations)
        return p_z

    def EM_fit(self, max_iter: int=100, delta_theta: float=1e-4):
        i = 0
        delta = delta_theta + 1
        while i < max_iter and delta > delta_theta:
            delta = self.EM_step()
            i += 1

        if i >= max_iter:
            print("WARN: Max iterations ({0}) reached during EM fit.".format(i))

    def _expectation_step(self):
        # Expectation (weight each observation componenent by likeihood with current parameters)
        self.p_z = self.data_likelihood()*self.component_weights
        self.p_z /= self.p_z.sum(axis=1, keepdims=True)
        
    def EM_step(self):
        self._expectation_step()
        
        # Maximisation
        cluster_mass = self.p_z.sum(axis=0)
        new_weights = cluster_mass/self.observations.shape[0]
        delta_theta = np.abs((self.component_weights - new_weights)).max()
        self.component_weights = new_weights
        for j, c in enumerate(self.components):
            c.maximise_likelihood(self.observations, self.p_z[:,[j]], cluster_mass[j])

        return delta_theta

    def plot_init(self, ax, plot_observations=False, color=None, **kwargs):
        self._artists = []
        self._plot_observations = plot_observations
        colour_array = make_rgb_array(self.n_components, color)
        if self._plot_observations:
            self._expectation_step()
            self._observation_plotter = WeightedData(ax, self.observations, weights=self.p_z, colours=colour_array, s=2)
            self._artists.extend(self._observation_plotter.get_artists())

        for c, rgb in zip(self.components, colour_array):
            self._artists.extend(c.create_plot_object(ax, color=rgb, **kwargs))
        return self._artists
    
    def plot_update(self):
        if self._plot_observations:
            self._observation_plotter.update(self.observations, self.p_z)

        for c in self.components:
            c.update_plot_object()
        return self._artists
        
        
class Component:
    
    def __init__(self, n_dim:int = 2) -> None:
        self.n_dim = n_dim
        self.distribution = self._init_distribution()

    def _init_distribution(self) -> None:
        # Method for sampling new parameters for this component
        raise NotImplementedError("The base Component class is just a placeholder, please use a subclass (e.g. GaussianComponent)")

    def likelihood(self, z) -> float:
        return self.distribution.pdf(z)  
    
    def sample(self, n: int=1):
        raise NotImplementedError("The base Component class is just a placeholder, please use a subclass (e.g. GaussianComponent)")
    
    def maximise_likelihood(self, z, p_z, mass: float) -> None:
        raise NotImplementedError("The base Component class is just a placeholder, please use a subclass (e.g. GaussianComponent)")
    
    def create_plot_object(self, ax):
        raise NotImplementedError("The base Component class is just a placeholder, please use a subclass (e.g. GaussianComponent)")

    def update_plot_object(self):
        raise NotImplementedError("The base Component class is just a placeholder, please use a subclass (e.g. GaussianComponent)")


class GaussianComponent(Component):

    def _init_distribution(self):
        # We will sample from Normal(0, 1) for mean and Wishart(1, n) for covariance
        mean = np.random.normal(0, 3, size=self.n_dim)
        covariance =  wishart.rvs(self.n_dim, np.identity(self.n_dim), 1)
        return multivariate_normal(mean, covariance)
    
    def maximise_likelihood(self, z, p_z, mass) -> None:
        imass = 1/mass
        self.distribution.mean = imass*(p_z*z).sum(axis=0)
        d = z - self.distribution.mean
        self.distribution.cov = imass*np.matmul(d.T, d*p_z)

    def sample(self, n: int = 1):
        return self.distribution.rvs(size=n)
        
    def create_plot_object(self, ax, n_points=101, ellipse_masses=[0.68, 0.95], **kwargs):
        if self.n_dim == 1:
            sigma = np.sqrt(self.distribution.cov)
            xx = np.linspace(self.distribution.mean-3*sigma, self.distribution.mean + 3*sigma, n_points)
            yy = self.likelihood(xx)
            self._artists = ax.plot(xx, yy, **kwargs)

        elif self.n_dim == 2:
            self._ellipses = CovarianceEllipses(ax, self.distribution.mean, self.distribution.cov, ellipse_masses=ellipse_masses, **kwargs)
            self._artists = self._ellipses.get_artists()
        else:
            raise NotImplementedError('Dimension > 2 plotting not supported')
        return self._artists
        
    def update_plot_object(self):
        if self.n_dim == 1:
            yy = self.likelihood(self._artists.get_xdata())
            self._artists.set_ydata(yy)
        elif self.n_dim == 2:
            self._ellipses.update(self.distribution.mean, self.distribution.cov)
        else:
            raise NotImplementedError('Dimension > 2 plotting not supported')
        return self._artists 