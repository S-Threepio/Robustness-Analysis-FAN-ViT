import torch
from tqdm import tqdm  # Displays a progress bar
import torch.nn.functional as F


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


def testFGSM(model, trad_model, device, test_loader, epsilon):

    # Accuracy counter
    adversarial_working_on_trad = 0
    adversarial_working_on_fan = 0
    # Loop over all examples in test set
    for data, target in tqdm(test_loader):

        if adversarial_working_on_trad == 100:
            break
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = trad_model(data)
        # print("shape of the prediction : ",output.shape)
        init_pred = output.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability

        # print(init_pred)
        # print("shape of the prediction : ",init_pred.shape)

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        trad_model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

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

    # Calculate final accuracy for this epsilon
    final_transferability = adversarial_working_on_fan / float(
        adversarial_working_on_trad
    )
    print(
        "\n\n transferability for FGSM attack = adversarial examples working on FAN vits/adversarial examples working on traditional vits ={}/{} = {}".format(
            adversarial_working_on_fan,
            adversarial_working_on_trad,
            final_transferability,
        )
    )
