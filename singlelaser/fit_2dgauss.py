''' Fit a 2D Gaussian to data '''

import numpy as np
import scipy.optimize

class Fit2DGauss:

    def gaussian(self, m, height, x0, y0, sig_x, sig_y, rotation):
        """Returns a gaussian function with the given parameters"""
        x, y = m
        x0, y0 = float(x0), float(y0)
        sig_x, sig_y = float(sig_x), float(sig_y)
        
        theta = np.deg2rad(rotation)
        
        a = (np.cos(theta)**2)/(2*sig_x**2) + (np.sin(theta)**2)/(2*sig_y**2)
        b = -(np.sin(2*theta))/(4*sig_x**2) + (np.sin(2*theta))/(4*sig_y**2)
        c = (np.sin(theta)**2)/(2*sig_x**2) + (np.cos(theta)**2)/(2*sig_y**2)
        g = height * np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))
        
        return g.ravel()
    
    def moments(self, data):
        """Returns (height, x, y, width_x, width_y)
            the gaussian parameters of a 2D distribution by calculating its
            moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = np.nan_to_num((X*data).sum()/total)
        y = np.nan_to_num((Y*data).sum()/total)
        col = data[:, int(y)]
        width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, width_x, width_y, 0.0

    def fitgaussian(self, data, params=None, bounds=(-np.inf, np.inf)):
        """Returns (height, x, y, width_x, width_y)
            the gaussian parameters of a 2D distribution found by a fit"""
        if params is None: params = self.moments(data)

        x = np.linspace(0,data.shape[0]+1,data.shape[0])
        y = np.linspace(0,data.shape[1]+1,data.shape[1])
        x, y = np.meshgrid(x, y)
        
        p, pcov = scipy.optimize.curve_fit(self.gaussian, (x, y), data.ravel(), p0=params, bounds=bounds)
        
        # Convert fit results to sane human-understandable numbers
        # Because of squares and infinite rotation, some widths can be -ve and angles any size
        # We could implement bounds but it's faster not to so we clean up here
        p[3], p[4], p[5] = np.abs(p[3]), np.abs(p[4]), p[5] % 360
        
        return p
