from generalized_logistic import GeneralizedLogistic
import torch


def generalized_logistic_test():
    """
    Provides Unit tests for the GeneralizedLogistic autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL1 (float): the  error tolerance for the forward mode. If the error >= TOL1, is_correct is false
    TOL2 (float): The error tolerance for the backward mode
    DELTA (float): The difference parameter for the finite differences computation
    X (Tensor): size (48 x 2) of inputs
    L, U, and G (floats): The parameter values necessary to compute the hyperbolic tangent (tanH) using
                        GeneralizedLogistic
    Returns:
    -------
    is_correct (boolean): True if and only if GeneralizedLogistic passes all unit tests
    err (Dictionary): with the following keys
                        1. y (float): The error between the forward direction and the results of pytorch's tanH
                        2. dzdx (float): the error between the analytical and numerical gradients w.r.t X
                        3. dzdl (float): ... w.r.t L
                        4. dzdu (float): ... w.r.t U
                        5. dzdg (float): .. w.r.t G
     Note
     -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%%% DO NOT EDIT BELOW %%%
    dataset = torch.load("generalized_logistic_test.pt")
    X = dataset["X"]
    L = dataset["L"]
    U = dataset["U"]
    G = dataset["G"]
    TOL1 = dataset["TOL1"]
    TOL2 = dataset["TOL2"]
    DELTA = dataset["DELTA"]
    generalized_logistic = GeneralizedLogistic.apply
    # %%%  DO NOT EDIT ABOVE %%%

    y = generalized_logistic(X,L,U,G)
    z = y.mean()
    z.backward()

    # CHECK DICT ###############################
    err = {'dzdx':[],'dzdl': [],'dzdu':[],'dzdg':[],'y':[]}

    # CHECK FORWARD PASS #######################

    y_err = abs(generalized_logistic(X,L,U,G) - torch.tanh(X)).max()
    err['y'] = y_err

    # z = y.mean()
    # gradcheck(GeneralizedLogistic.apply, (X,L,U,G), eps=DELTA, atol=TOL2)


    # NUMERICALLY APPROXIMATED GRADIENT (finite difference)

    dz_dy = torch.autograd.grad(z, y) # this is 48 x 72

    # FINITE DIFF CALCS ##############################

    is_correct = True

    if y_err > TOL1:
        is_correct = False


    # X gradient #####################################
    with torch.no_grad():

        num_x = torch.zeros(X.shape)

        # for each row
        for b in range(0, X.shape[1]):

            # for each column
            for a in range(0, X.shape[0]):
                x_clone = X.clone()
                x_plus = X.clone();
                x_minus = X.clone()
                zers = torch.zeros(X.shape)

                # make x_plus and x_minus
                x_minus[a, b] = x_minus[a, b] - DELTA
                x_plus[a, b]  = x_plus[a, b] + DELTA

                # calculate gradient
                grad_x = dz_dy[0] * (
                (generalized_logistic(x_plus, L, U, G) - generalized_logistic(x_minus, L, U, G)) / (DELTA * 2))
                num_x[a, b] = grad_x.sum()

        diff_x = X.grad - num_x
        abs_diff_x = abs(diff_x)

        err['dzdx'] = abs_diff_x.max()

        if err['dzdx'] > TOL2:
            is_correct = False

        # L gradient #####################################
        with torch.no_grad():

            num_l = torch.zeros(L.shape)

            # for each row
            #for b in range(0, L.shape[1]):

            # for each column
            # for a in range(0, L.shape[0]):
            #     l_clone = L.clone()
            l_plus  = L.clone();
            l_minus = L.clone()
            zers = torch.zeros(L.shape)

            # make x_plus and x_minus
            l_minus = l_minus - DELTA
            l_plus  = l_plus + DELTA

            # calculate gradient
            grad_l = dz_dy[0] * (
                (generalized_logistic(X, l_plus, U, G) - generalized_logistic(X, l_minus, U, G)) / (DELTA * 2))
            num_l = grad_l.sum()

            diff_l = L.grad - num_l
            abs_diff_l = abs(diff_l)

            err['dzdl'] = abs_diff_l.max()

            if err['dzdl'] > TOL2:
                is_correct = False

        # U gradient #####################################
        with torch.no_grad():

            num_u = torch.zeros(U.shape)

            # for each row
            # for b in range(0, u.shape[1]):

            # for each column
            # for a in range(0, u.shape[0]):
            #     u_clone = u.clone()
            u_plus  = U.clone();
            u_minus = U.clone()
            zers = torch.zeros(U.shape)

            # make x_plus and x_minus
            u_minus = u_minus - DELTA
            u_plus  = u_plus + DELTA

            # calculate gradient
            grad_u = dz_dy[0] * (
                (generalized_logistic(X, L, u_plus, G) - generalized_logistic(X, L, u_minus, G)) / (
                DELTA * 2))
            num_u = grad_u.sum()

            diff_u = U.grad - num_u
            abs_diff_u = abs(diff_u)

            err['dzdu'] = abs_diff_u.max()

            if err['dzdu'] > TOL2:
                is_correct = False

        # G gradient #####################################
        with torch.no_grad():

            num_g = torch.zeros(G.shape)

            # for each row
            # for b in range(0, g.shape[1]):

            # for each column
            # for a in range(0, g.shape[0]):
            #     g_clone = g.clone()
            g_plus  = G.clone();
            g_minus = G.clone()
            zers = torch.zeros(G.shape)

            # make x_plus and x_minus
            g_minus = g_minus - DELTA
            g_plus = g_plus + DELTA

            # calculate gradient
            grad_g = dz_dy[0] * (
                (generalized_logistic(X, L, U, g_plus) - generalized_logistic(X, L, U, g_minus)) / (
                    DELTA * 2))
            num_g = grad_g.sum()

            diff_g = G.grad - num_g
            abs_diff_g = abs(diff_g)

            err['dzdg'] = abs_diff_g.max()

            if err['dzdg'] > TOL2:
                is_correct = False

    ############################################################

    # Final check
    if torch.autograd.gradcheck(GeneralizedLogistic.apply, (X, L, U, G), eps=DELTA, atol=TOL2) == False:
        is_correct = False

    # save
    torch.save([is_correct, err], 'generalized_logistic_test_results.pt')

    return is_correct, err


if __name__ == '__main__':
    test_passed, errors = generalized_logistic_test()
    assert test_passed
    print(errors)
