import torch
from tqdm import tqdm  # Displays a progress bar
import torch.nn.functional as F
from .utils import *


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def testFGSM(model, device, test_loader, epsilon):

    # Accuracy counter
    incorrectly_predicted_outputs_after_attack = 0
    correctly_predicted_outputs_before_attack = 0
    adv_examples = []
    l2_fgsm = 0
    linf_fgsm = 0
    # Loop over all examples in test set
    for data, target in tqdm(test_loader):
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        # print("shape of the prediction : ",output.shape)
        init_pred = output.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability

        # print(init_pred)
        # print("shape of the prediction : ",init_pred.shape)

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        correctly_predicted_outputs_before_attack += 1

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        l2_fgsm += l2_norm(perturbed_data, data)
        linf_fgsm += linf(perturbed_data, data)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability
        if final_pred.item() != target.item():
            incorrectly_predicted_outputs_after_attack += 1
            break
            # Save some adv examples for visualization later
            if len(adv_examples) < 10:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    # Calculate final accuracy for this epsilon
    final_acc = incorrectly_predicted_outputs_after_attack / float(
        correctly_predicted_outputs_before_attack
    )
    print(
        "FGSM Attack\nEpsilon: {}\nAttack Accuracy = (incorrectly predicted labels after attack) / (correctly predicted labels before attack) = {} / {} = {}\n\n".format(
            epsilon,
            incorrectly_predicted_outputs_after_attack,
            correctly_predicted_outputs_before_attack,
            final_acc,
        )
    )
    print(
        "The average L2 distance for FGSM is {}\nThe average Linf distance for FGSM is {}".format(
            l2_fgsm / correctly_predicted_outputs_before_attack,
            linf_fgsm / correctly_predicted_outputs_before_attack,
        )
    )

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
