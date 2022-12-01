import torch
from tqdm import tqdm  # Displays a progress bar
import torch.nn.functional as F

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


def testPGD(model, trad_model, device, test_loader, epsilon, steps=10):
    # Accuracy counter
    adversarial_working_on_trad = 0
    adversarial_working_on_fan = 0
    for data, target in tqdm(test_loader):
        if adversarial_working_on_trad == 100:
            break

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = trad_model(data)
        init_pred = output.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        perturbed_data = pgd_attack(
            trad_model, data.to(device), target.to(device), epsilon, steps
        )

        # Re-classify the perturbed image
        output = trad_model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability
        if final_pred.item() != target.item():
            adversarial_working_on_trad += 1

            fan_output = model(perturbed_data)
            fan_pred = fan_output.max(1, keepdim=True)[1]
            if fan_pred.item() != target.item():
                adversarial_working_on_fan += 1

            # Save some adv examples for visualization later
            # if len(adv_examples) < 10:
            #     adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            #     adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_transferability = adversarial_working_on_fan / float(
        adversarial_working_on_trad
    )
    print(
        "Steps ",
        steps,
        "\n"
        "\n\n transferability for PGD attack = adversarial examples working on FAN vits/adversarial examples working on traditional vits = {}/{} = {}".format(
            adversarial_working_on_fan,
            adversarial_working_on_trad,
            final_transferability,
        ),
    )
