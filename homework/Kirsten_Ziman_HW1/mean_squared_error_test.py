from mean_squared_error import MeanSquaredError
import torch


def mean_squared_error_test():
    """
     Unit tests for the MeanSquaredError autograd Function.

    PROVIDED CONSTANTS
    ------------------
    TOL (float): the absolute error tolerance for the backward mode. If any error is equal to or
                greater than TOL, is_correct is false
    DELTA (float): The difference parameter for the finite difference computation
    X1 (Tensor): size (48 x 2) denoting 72 example inputs each with 2 features
    X2 (Tensor): size (48 x 2) denoting the targets

    Returns
    -------
    is_correct (boolean): True if and only if MeanSquaredError passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx1 (float): the  error between the analytical and numerical gradients w.r.t X1
                    2. dzdx2 (float): The error between the analytical and numerical gradients w.r.t X2
    Note
    -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%
    dataset = torch.load("mean_squared_error_test.pt")
    X1 = dataset["X1"]
    X2 = dataset["X2"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    mean_squared_error = MeanSquaredError.apply
    # %%% DO NOT EDIT ABOVE %%%

    ##########################################################

    # Compare analytical gradient w/ numerically approximated gradient

    # ANALYTICAL GRADIENT
    y = mean_squared_error(X1,X2)
    z = y.mean() # this is our simple function, J(y; theta)
    z.backward()

    # NUMERICALLY APPROXIMATED GRADIENT (finite difference)

    dz_dy = torch.autograd.grad(z, y) # this is 48 x 72

    # FINITE DIFF CALCS ##############################

    is_correct = True
    err = {'dzdx1': [], 'dzdx2': []}
    #
    # x1_clone = X1.clone()
    # x2_clone = X2.clone()

    # X1 gradient #####################################
    with torch.no_grad():

        num_x1 = torch.zeros(X1.shape)

        # for each row
        for b in range(0, X1.shape[1]):

            # for each column
            for a in range(0, X1.shape[0]):

                x1_clone = X1.clone()
                x1_plus  = X1.clone(); x1_minus = X1.clone()
                zers    = torch.zeros(X1.shape)

                # make x_plus and x_minus
                x1_minus[a,b] = x1_minus[a,b]-DELTA
                x1_plus[a,b]  = x1_plus[a,b]+DELTA

                # calculate gradient
                grad_x1 = dz_dy[0] * ((mean_squared_error(x1_plus, X2) - mean_squared_error(x1_minus, X2)) / (DELTA * 2))
                num_x1[a,b] = grad_x1.sum()

        diff_x1 = X1.grad - num_x1
        abs_diff_x1 = abs(diff_x1)

        err['dzdx1'] = abs_diff_x1.max()

        if err['dzdx1'] > TOL:
            is_correct = False

        # X2 gradient #####################################
        with torch.no_grad():

            num_x2 = torch.zeros(X2.shape)

            # for each row
            for b in range(0, X2.shape[1]):

                # for each column
                for a in range(0, X2.shape[0]):
                    x2_clone = X2.clone()
                    x2_plus = X2.clone();
                    x2_minus = X2.clone()
                    zers = torch.zeros(X2.shape)

                    # make x_plus and x_minus
                    x2_minus[a, b] = x2_minus[a, b] - DELTA
                    x2_plus[a, b]  = x2_plus[a, b] + DELTA

                    # calculate gradient
                    grad_x2 = dz_dy[0] * (
                    (mean_squared_error(X1,x2_plus) - mean_squared_error(X1, x2_minus)) / (DELTA * 2))
                    num_x2[a, b] = grad_x2.sum()

            diff_x2 = X2.grad - num_x2
            abs_diff_x2 = abs(diff_x2)

            err['dzdx2'] = abs_diff_x2.max()

            if err['dzdx2'] > TOL:
                is_correct = False

    ############################################################

    # Final check
    if torch.autograd.gradcheck(MeanSquaredError.apply, (X1, X2), eps=DELTA, atol=TOL) == False :
        is_correct = False

    # save
    torch.save([is_correct, err],'mean_squared_error_test_results.pt')

    ############################################################
    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = mean_squared_error_test()
    assert tests_passed
    print(errors)
