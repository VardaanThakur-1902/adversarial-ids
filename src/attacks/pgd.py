import torch


def pgd_attack(
    model,
    x,
    y,
    epsilon,
    alpha,
    num_steps,
    loss_fn
):
    """
    Generate PGD adversarial examples
    under an L-infinity constraint.

    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network.

    x : torch.Tensor
        Original input samples.

    y : torch.Tensor
        True labels.

    epsilon : float
        Maximum L-infinity perturbation.

    alpha : float
        Step size for each PGD iteration.

    num_steps : int
        Number of attack iterations.

    loss_fn : callable
        Loss function.

    Returns
    -------
    torch.Tensor
        PGD adversarial examples.
    """

    # Start from the original input
    x_original = x.clone().detach()

    x_adv = x_original.clone().detach()

    for _ in range(num_steps):

        x_adv.requires_grad = True

        # Forward pass
        outputs = model(x_adv)

        # Calculate loss
        loss = loss_fn(
            outputs,
            y
        )

        # Clear previous gradients
        model.zero_grad()

        # Calculate gradient
        loss.backward()

        # Gradient-sign update
        gradient = x_adv.grad.sign()

        x_adv = (
            x_adv +
            alpha * gradient
        )

        # Project back into L-infinity epsilon ball
        perturbation = (
            x_adv - x_original
        )

        perturbation = torch.clamp(
            perturbation,
            min=-epsilon,
            max=epsilon
        )

        x_adv = (
            x_original +
            perturbation
        ).detach()

    return x_adv