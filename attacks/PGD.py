import torch
from tqdm import tqdm  # Displays a progress bar
import torch.nn.functional as F
from .utils import *

PGD_step_wise_accuracy = []


def pgd_attack(model, image, label, epsilon, steps=10, alpha=2 / 255):

    ori_image = image
    for i in range(steps):
        image.requires_grad = True
        output = model(image)
        # Calculate the loss
        loss = F.nll_loss(output, label)
        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = image.grad.data

        adv_image = image + alpha * data_grad.sign()
        eta = torch.clamp(adv_image - ori_image, min=-epsilon, max=epsilon)
        image = torch.clamp(ori_image + eta, min=0, max=1).detach_()
    return image


def testPGD(model, device, test_loader, epsilon, steps=10):
    # Accuracy counter
    incorrectly_predicted_outputs_after_attack = 0
    correctly_predicted_outputs_before_attack = 0
    adv_examples = []

    l2_pgd = 0
    linf_pgd = 0

    dataCount = 0
    for data, target in tqdm(test_loader):
        if dataCount == 1000:
            break
        dataCount += 1

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        correctly_predicted_outputs_before_attack += 1

        perturbed_data = pgd_attack(
            model, data.to(device), target.to(device), epsilon, steps
        )

        l2_pgd += l2_norm(perturbed_data, data)
        linf_pgd += linf(perturbed_data, data)

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
        "\n\nSteps: {}\nAttack Accuracy = (incorrectly predicted labels after attack) / (correctly predicted labels before attack) =  {} / {} = {}".format(
            steps,
            incorrectly_predicted_outputs_after_attack,
            correctly_predicted_outputs_before_attack,
            final_acc,
        )
    )
    print(
        "The average L2 distance for PGD is {}\nThe average Linf distance for PGD is {}".format(
            l2_pgd / correctly_predicted_outputs_before_attack,
            linf_pgd / correctly_predicted_outputs_before_attack,
        )
    )

    PGD_step_wise_accuracy.append(final_acc)
    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
