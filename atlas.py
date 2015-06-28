import numpy as np


def atlas_for(v, atlas=None):
    """
    Get the atlas for the object v.
    First, if the atlas keyword parameter is not None then it is returned.
    Next, type(v).Atlas is defined then it is returned.
    Next, if v is a scalar then ScalarAtlas is returned.
    Otherwise, CartesianAtlas is returned.
    """
    if atlas is None:
        try:
            return type(v).Atlas
        except AttributeError:
            if np.isscalar(v):
                return ScalarAtlas
            else:
                return CartesianAtlas
    else:
        return atlas


class ScalarAtlas(object):
    """
    Represents an atlas for real numbers.
    """
    @classmethod
    def dof(cls, v):
        return 1

    @classmethod
    def perturb(cls, v, tangent):
        return v + tangent[0]

    @classmethod
    def displacement(cls, v1, v2):
        """Get a vector v such that perturb(v1, v) = v2."""
        return np.array([v2 - v1])


class CartesianAtlas(object):
    """
    Represents an atlas for finite-dimensional vector spaces.
    """
    @classmethod
    def dof(cls, v):
        return len(v)

    @classmethod
    def perturb(cls, v, tangent):
        return (v + tangent).copy()

    @classmethod
    def displacement(cls, v1, v2):
        """Get a vector v such that perturb(v1, v) = v2."""
        return (v2 - v1).copy()


class HomogeneousAtlas(object):
    """
    Represents an atlas for finite-dimensional projective spaces.
    """
    @classmethod
    def dof(cls, v):
        return len(v) - 1

    @classmethod
    def perturb(cls, v, delta):
        return np.r_[v[:-1] + delta, v[-1]]

    @classmethod
    def displacement(cls, v1, v2):
        v1, w1 = v1[:-1], v1[-1]
        v2, w2 = v2[:-1], v2[-1]
        return (w1 * v2 - w2 * v1) / w2
