from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np


def gmm_sample(n_samples, mix_coeffs, mean, cov):
    z = np.random.multinomial(n_samples, mix_coeffs)
    samples = np.zeros(shape=[n_samples, len(mean[0])])
    i_start = 0
    for i in range(len(mix_coeffs)):
        i_end = i_start + z[i]
        samples[i_start:i_end, :] = np.random.multivariate_normal(
            mean=np.array(mean)[i,:],
            cov=np.diag(np.array(cov)[i,:]),
            size=z[i])
        i_start = i_end
    return samples

# circle
def sampler1(n_samples):
    n_mixs = 8
    radius = 2.0
    std = 0.02
    thetas = np.linspace(0, 2*np.pi, n_mixs+1)[:n_mixs]
    mean = tuple(zip(radius * np.sin(thetas), radius * np.cos(thetas)))
    cov = tuple([(std, std)] * n_mixs)
    mix_coeffs = tuple([1./n_mixs] * n_mixs)
    samples = gmm_sample(n_samples, mix_coeffs, mean, cov)
    return samples

# grid
def sampler2(n_samples):
    n_rows, n_cols = 3, 3
    n_mixs = n_rows * n_cols
    interval = 2.0
    std = 0.02
    x0, y0 = -interval*(n_cols-1)/2., -interval*(n_rows-1)/2.
    mean = tuple((x0+j*interval, y0+i*interval) for i in range(n_rows) for j in range(n_cols))
    cov = tuple([(std, std)] * n_mixs)
    mix_coeffs = tuple([1./n_mixs] * n_mixs)
    samples = gmm_sample(n_samples, mix_coeffs, mean, cov)
    return samples

if __name__ == '__main__':
    import toy_experiments.visualizer as vis
    # s = sampler1(100)
    s = sampler2(100)
    fig, ax, plt = vis.display_scatter(s)
    fig.tight_layout()
    plt.show()
    #fig.savefig("\{}.png".format('example'))