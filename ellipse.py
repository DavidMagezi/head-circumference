#!/bin/python

# Most of the code below has been sourced from Christian Hill 
# https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

# Modifications by David A. Magezi (2023)

# %%
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipe

class Ellipse:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def fit(self):
        coeffs = self.fit_ellipse()
        x0, y0, ap, bp, e, phi = self.cart_to_pol(coeffs)
        x_fitted, y_fitted = self.get_ellipse_pts((x0, y0, ap, bp, e, phi))
        self.ap = ap
        self.bp = bp
        self.e = e
        return (x_fitted,y_fitted)

    def fit_ellipse(self):
        """

        Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
        the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
        arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

        Based on the algorithm of Halir and Flusser, "Numerically stable direct
        least squares fitting of ellipses'.


        """

        x = self.x
        y = self.y

        D1 = np.vstack([x**2, x*y, y**2]).T
        D2 = np.vstack([x, y, np.ones(len(x))]).T
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2
        T = -np.linalg.inv(S3) @ S2.T
        M = S1 + S2 @ T
        C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        M = np.linalg.inv(C) @ M
        eigval, eigvec = np.linalg.eig(M)
        con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
        ak = eigvec[:, np.nonzero(con > 0)[0]]
        return np.concatenate((ak, T @ ak)).ravel()


    def cart_to_pol(self,coeffs):
        """

        Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
        ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
        The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
        ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
        respectively; e is the eccentricity; and phi is the rotation of the semi-
        major axis from the x-axis.

        """

        # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
        # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
        # Therefore, rename and scale b, d and f appropriately.
        a = coeffs[0]
        b = coeffs[1] / 2
        c = coeffs[2]
        d = coeffs[3] / 2
        f = coeffs[4] / 2
        g = coeffs[5]

        den = b**2 - a*c
        if den > 0:
            raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                             ' be negative!')

        # The location of the ellipse centre.
        x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

        num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        fac = np.sqrt((a - c)**2 + 4*b**2)
        # The semi-major and semi-minor axis lengths (these are not sorted).
        ap = np.sqrt(num / den / (fac - a - c))
        bp = np.sqrt(num / den / (-fac - a - c))

        # Sort the semi-major and semi-minor axis lengths but keep track of
        # the original relative magnitudes of width and height.
        width_gt_height = True
        if ap < bp:
            width_gt_height = False
            ap, bp = bp, ap
        self.width_gt_height = width_gt_height

        # The eccentricity.
        r = (bp/ap)**2
        if r > 1:
            r = 1/r
        e = np.sqrt(1 - r)

        # The angle of anticlockwise rotation of the major-axis from x-axis.
        if b == 0:
            phi = 0 if a < c else np.pi/2
        else:
            phi = np.arctan((2.*b) / (a - c)) / 2
            if a > c:
                phi += np.pi/2
        if not width_gt_height:
            # Ensure that phi is the angle to rotate to the semi-major axis.
            phi += np.pi/2
        phi = phi % np.pi

        return x0, y0, ap, bp, e, phi


    def get_ellipse_pts(self,params, npts=100, tmin=0, tmax=2*np.pi):
        """
        Return npts points on the ellipse described by the params = x0, y0, ap,
        bp, e, phi for values of the parametric variable t between tmin and tmax.

        """

        x0, y0, ap, bp, e, phi = params
        # A grid of the parametric variable, t.
        t = np.linspace(tmin, tmax, npts)
        x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
        y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
        return x, y

    def estimate_circumference(self,pixel_size,original_dimensions,current_dimensions):
        # Modified from scipython.com/book/chapter-8-scipy
        if self.width_gt_height:
            new_ap = self.ap * (original_dimensions[0]/current_dimensions[0])
            new_bp = self.bp * (original_dimensions[1]/current_dimensions[1])
        else:
            new_ap = self.ap * (original_dimensions[1]/current_dimensions[1])
            new_bp = self.bp * (original_dimensions[0]/current_dimensions[0])

        if new_ap < new_bp:
            new_ap, new_bp = new_bp, new_ap

        new_e = math.sqrt(1-(pow(new_bp,2)/pow(new_ap,2)))
        estimated_circumference = 4 * new_ap * ellipe (new_e*new_e)
        print(pixel_size)
        return (estimated_circumference * pixel_size)
