from fully_connected import FullyConnected
import torch


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


    # FINITE DIFF CALCS ##############################

    is_correct = True
    err = {'dzdx':[], 'dzdw':[], 'dzdb':[]}

    # X gradient #####################################
    with torch.no_grad():

        num_x = torch.zeros(X.shape)

        # for each row
        for b in range(0, X.shape[1]):

            # for each column
            for a in range(0, X.shape[0]):

                x_clone = X.clone()
                x_plus  = X.clone(); x_minus = X.clone()
                zers    = torch.zeros(X.shape)

                # make x_plus and x_minus
                x_minus[a,b] = x_minus[a,b]-DELTA
                x_plus[a,b]  = x_plus[a,b]+DELTA

                # calculate gradient
                grad_x = dz_dy[0] * ((full_connected(x_plus, W, B) - full_connected(x_minus, W, B)) / (DELTA * 2))
                num_x[a,b] = grad_x.sum()

        diff_x = X.grad - num_x
        abs_diff_x = abs(diff_x)

        err['dzdx'] = abs_diff_x.max()

        if err['dzdx'] > TOL:
            is_correct = False

    # W gradient #####################################
    with torch.no_grad():

        num_w = torch.zeros(W.shape)

        # for each row
        for b in range(0, W.shape[1]):

            # for each column
            for a in range(0, W.shape[0]):
                w_clone = W.clone()
                w_plus = W.clone();
                w_minus = W.clone()
                zers = torch.zeros(W.shape)

                # make x_plus and x_minus
                w_minus[a, b] = w_minus[a, b] - DELTA
                w_plus[a, b] = w_plus[a, b] + DELTA

                # calculate gradient
                grad_w = dz_dy[0] * ((full_connected(X, w_plus, B) - full_connected(X, w_minus, B)) / (DELTA * 2))
                num_w[a, b] = grad_w.sum()

        diff_w = W.grad - num_w
        abs_diff_w = abs(diff_w)

        err['dzdw'] = abs_diff_w.max()

        if err['dzdw'] > TOL:
            is_correct = False

    # B gradient #####################################
    with torch.no_grad():

        num_b = torch.zeros(B.shape)

        # for each row
        for b in range(0, B.shape[0]):

            b_clone = B.clone()
            b_plus = B.clone();
            b_minus = B.clone()
            zers = torch.zeros(B.shape)

            # make x_plus and x_minus
            b_minus[b] = b_minus[b] - DELTA
            b_plus[b] = b_plus[b] + DELTA

            # calculate gradient
            grad_b = dz_dy[0] * ((full_connected(X, W, b_plus) - full_connected(X, W, b_minus)) / (DELTA * 2))
            num_b[b] = grad_b.sum()

        diff_b = B.grad - num_b
        abs_diff_b = abs(diff_b)

        err['dzdb'] = abs_diff_b.max()

        if err['dzdb'] > TOL:
            is_correct = False

    ############################################################

    # Final check
    if torch.autograd.gradcheck(FullyConnected.apply, (X, W, B), eps=DELTA, atol=TOL) == False :
        is_correct = False

    # save
    torch.save([is_correct, err],'fully_connected_test_results.pt')

    ############################################################

    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = fully_connected_test()
    assert tests_passed
    print(errors)
