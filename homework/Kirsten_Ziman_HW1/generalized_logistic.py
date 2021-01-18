import torch


class GeneralizedLogistic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l, u, g):
        """
        Computes the generalized logistic function

        Arguments
        ---------
        ctx: A PyTorch context object
        x: (Tensor) of size (T x n), the input features
        l, u, and g: (scalar tensors) representing the generalized logistic function parameters.

        Returns
        -------
        y: (Tensor) of size (T x n), the outputs of the generalized logistic operator

        """
        e = 2.718281828459045

        ctx.save_for_backward(x,l,u,g)
        y = l+((u-l)/(1+e**(-g*x)))

        return y

    @staticmethod
    def backward(ctx, dzdy):
        """
        back-propagate the gradients with respect to the inputs

        Arguments
        ----------
        ctx: a PyTorch context object
        dzdy (Tensor): of size (T x n), the gradients with respect to the outputs y

        Returns
        -------
        dzdx (Tensor): of size (T x n), the gradients with respect to x
        dzdl, dzdu, and dzdg: the gradients with respect to the generalized logistic parameters
        """

        e = 2.718281828459045

        x,l,u,g = ctx.saved_tensors
        dzdx = dzdy * ((g * (u-l) * e**(-g*x)) / (1 + e**(-g*x) )**2)
        dzdl = dzdy * (1/(e**(g*x)+1))
        dzdu = dzdy * (1/(e**(-g*x)+1))
        dzdg = dzdy * ((x * (u-l) * e**(-g*x)) / (1 + e**(-g*x) )**2)

        return dzdx, dzdl, dzdu, dzdg
