import torch


def fgsm_attack(
    model,
    x,
    y,
    epsilon,
    loss_fn
):
    """
    Generate FGSM adversarial examples.

    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network.

    x : torch.Tensor
        Input samples in standardized feature space.

    y : torch.Tensor
        True binary labels.

    epsilon : float
        Maximum L-infinity perturbation.

    loss_fn : callable
        Loss function used to calculate the gradient.

    Returns
    -------
    torch.Tensor
        Adversarial examples.
    """

    x_adv = x.clone().detach()
    x_adv.requires_grad = True

    # Forward pass
    outputs = model(x_adv)

    # Calculate loss
    loss = loss_fn(outputs, y)

    # Clear old gradients
    model.zero_grad()

    # Calculate gradient with respect to input
    loss.backward()

    # FGSM perturbation
    perturbation = epsilon * x_adv.grad.sign()

    # Create adversarial example
    x_adv = x_adv + perturbation

    return x_adv.detach()