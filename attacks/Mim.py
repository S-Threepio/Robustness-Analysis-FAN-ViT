import torch
from tqdm import tqdm  # Displays a progress bar
import torch.nn.functional as F
from .utils import *
import torchattacks


def testMiM(model, device, test_loader, epsilon, ip_steps):

    # Accuracy counter
    incorrectly_predicted_outputs_after_attack = 0
    correctly_predicted_outputs_before_attack = 0
    adv_examples = []

    l2_mim = 0
    linf_mim = 0

    attack = torchattacks.MIFGSM(model, eps=8 / 255, steps=ip_steps, decay=1.0)
    # Loop over all examples in test set
    dataCount = 0
    for data, target in tqdm(test_loader):
        if dataCount == 1000:
            break
        dataCount += 1  # Send the data and label to the device
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

        perturbed_data = attack(data, target)

        l2_mim += l2_norm(perturbed_data, data)
        linf_mim += linf(perturbed_data, data)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability
        if final_pred.item() != target.item():
            incorrectly_predicted_outputs_after_attack += 1

            # Save some adv examples for visualization later
            # if len(adv_examples) < 10:
            #     adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            #     adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    # Calculate final accuracy for this epsilon
    final_acc = incorrectly_predicted_outputs_after_attack / float(
        correctly_predicted_outputs_before_attack
    )
    print(
        "\n\nMIM Attack\nEpsilon: {}\nAttack Accuracy = (incorrectly predicted labels after attack) / (correctly predicted labels before attack) = {} / {} = {}\n\n".format(
            epsilon,
            incorrectly_predicted_outputs_after_attack,
            correctly_predicted_outputs_before_attack,
            final_acc,
        )
    )
    print(
        "The average L2 distance for MIM is {}\nThe average Linf distance for MIM is {}".format(
            l2_mim / correctly_predicted_outputs_before_attack,
            linf_mim / correctly_predicted_outputs_before_attack,
        )
    )
    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
