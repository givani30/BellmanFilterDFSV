def bellman_objective(params, y, filter):
    """
    Compute the Bellman objective function for the DFSV model.

    Parameters
    ----------
    params : DFSV_params
        Model parameters.
    y : np.ndarray
        Observed data.
    filter : DFSVBellmanFilter
        Bellman filter object.

    Returns
    -------
    float
        The Bellman objective value.
    """
    # run the bellman filter
    _, _, ll = filter.run(params, y)
    return ll
