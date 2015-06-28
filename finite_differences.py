import numpy as np

from atlas import atlas_for


def numeric_jacobian_colwise(f, x, atlas=None, output_atlas=None, h=1e-8):
    """
    Compute the jacobian of f at x using finite differences.
    See the documentation for `numeric_jacobian` for details. This function
    differs only in that it is a generator that yields each column of the numeric
    jacobian in turn.
    """
    y = f(x)

    input_atlas = atlas_for(x, atlas)
    output_atlas = atlas_for(y, output_atlas)

    xdof = input_atlas.dof(x)
    ydof = output_atlas.dof(y)
    for i in range(xdof):
        ei = (np.arange(xdof) == i).astype(int)
        yield output_atlas.displacement(f(input_atlas.perturb(x, -ei*h)),
                                        f(input_atlas.perturb(x, ei*h))) / (2. * h)


def numeric_jacobian(f, x, atlas=None, output_atlas=None, h=1e-8):
    """
    Compute the jacobian for f at x using finite differences.
    """
    return np.array(list(numeric_jacobian_colwise(f, x, atlas, output_atlas, h))).T
