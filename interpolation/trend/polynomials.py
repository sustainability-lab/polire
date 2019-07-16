"""File containing polynomials supported by Trend interpolation
Class.
"""
def _create_polynomial(order):
    if order is None: # custom function by the user
        return None

    elif order == 0:
        def func(X, a):
            return a

    elif order == 1:
        def func(X, a, b, c):
            x1, x2 = X
            return a + b*x1 + c*x2

    elif order == 2:
        def func(X, a, b, c, d, e, f):
            x1, x2 = X
            return a + b*x1 + c*x2 + d*(x1**2) + e*(x1**2) + f*x1*x2

    else:
        raise NotImplementedError\
            (f"{order} order polynomial needs to be defined manually")

    return func
