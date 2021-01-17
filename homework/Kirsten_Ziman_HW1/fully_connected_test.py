from fully_connected import FullyConnected
import torch
from torch.autograd import gradcheck


def fully_connected_test():
    """
    Provides Unit tests for the FullyConnected autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL (float): The error tolerance for the backward mode. If the error >= TOL, then is_correct is false
    DELTA (float): The difference parameter for the finite difference computations
    X (Tensor): of size (48 x 2), the inputs
    W (Tensor): of size (2 x 72), the weights
    B (Tensor): of size (72), the biases

    Returns
    -------
    is_correct (boolean): True if and only iff FullyConnected passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx: the error between the analytical and numerical gradients w.r.t X
                    2. dzdw (float): ... w.r.t W
                    3. dzdb (float): ... w.r.t B

    Note
    ----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%%
    dataset = torch.load("fully_connected_test.pt")
    X = dataset["X"]
    W = dataset["W"]
    B = dataset["B"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    full_connected = FullyConnected.apply
    # %%% DO NOT EDIT ABOVE

    ###########################################################


    # Compare analytical gradient w/ numerically approximated gradient

    # ANALYTICAL GRADIENT
    y = full_connected(X, W, B)
    z = y.mean() # this is our simple function, J(y; theta)
    z.backward()

    ana_x, ana_b, ana_w = X.grad, B.grad, W.grad

    # NUMERICALLY APPROXIMATED GRADIENT (finite difference)
    dz_dy = torch.autograd.grad(z, y) # this is 48 x 72


    # dzdx, dzdw, dzdb = FullyConnected.backward(dz_dy)


    with torch.no_grad():

        X_PLUS = X + DELTA; X_MINUS = X - DELTA
        B_PLUS = B + DELTA; B_MINUS = B - DELTA
        W_PLUS = W + DELTA; W_MINUS = W - DELTA

        num_x = dz_dy[0]*((full_connected(X_PLUS, W, B) - full_connected(X_MINUS, W, B)) / (DELTA * 2))
        num_b = dz_dy[0]*((full_connected(X, W, B_PLUS) - full_connected(X, W, B_MINUS)) / (DELTA * 2))
        num_w = dz_dy[0]*((full_connected(X, W_PLUS, B) - full_connected(X, W_MINUS, B)) / (DELTA * 2))

    e_x = dz_dy - num_x; e_x = e_x.max()
    e_w = dz_dy - num_w; e_w = e_w.max()
    e_b = dz_dy - num_b; e_b = e_b.max()

    ############################################################

    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = fully_connected_test()
    assert tests_passed
    print(errors)
