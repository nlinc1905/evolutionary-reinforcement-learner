def quadratic_fxn_fitness(params):
    """
    This function is like the environment, which takes a set of params and returns a reward.
    It is the function to be maximized, a.k.a. the fitness function or objective function.

    :param params: a 1D numpy array of parameters to pass to the environment or function
    """
    if len(params) != 3:
        raise Exception(f"This environment requires 3 parameters and {len(params)} were given.")
    x0 = params[0]
    x1 = params[1]
    x2 = params[2]
    # return a quadratic function as a simple example
    # optimal values will be 0, 1, and -2, giving a reward of 0
    return -(x0 ** 2 + 0.1 * (x1 - 1) ** 2 + 0.5 * (x2 + 2) ** 2)
