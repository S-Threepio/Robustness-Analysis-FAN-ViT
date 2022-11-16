import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
from FAN.models import fan
from timm.models import create_model
import json


with open("labels.txt") as f:
    data = f.read()
        
# reconstructing the data as a dictionary
ground_labels = json.loads(data)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model_path = './pretrained_models/fan_vit_tiny.pth.tar'
print("Loading datasets...")
imagenet_path = r'./val'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize,
			])
imagenet_data = datasets.ImageFolder(imagenet_path, transform=transform)
val_data_loader = DataLoader(
	imagenet_data,
	batch_size=1,
	shuffle=True,
	num_workers=0
)
print("Done!")

# #checking if validation data is loaded correctly or not
# examples = enumerate(val_data_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# print(batch_idx)
# for i in range(5):
#     plt.imshow(example_data[i][0])
#     plt.show()


PGD_step_wise_accuracy = []




# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
	# Collect the element-wise sign of the data gradient
	sign_data_grad = data_grad.sign()
	# Create the perturbed image by adjusting each pixel of the input image
	perturbed_image = image + epsilon*sign_data_grad
	# Adding clipping to maintain [0,1] range
	perturbed_image = torch.clamp(perturbed_image, 0, 1)
	# Return the perturbed image
	return perturbed_image

def testFGSM(model, device, test_loader, epsilon):

	# Accuracy counter
	incorrectly_predicted_outputs_after_attack = 0
	correctly_predicted_outputs_before_attack = 0
	adv_examples = []
	# Loop over all examples in test set
	for data, target in tqdm(test_loader):
		# Send the data and label to the device
		data, target = data.to(device), target.to(device)
		# Set requires_grad attribute of tensor. Important for Attack
		data.requires_grad = True

		# Forward pass the data through the model
		output = model(data)
		# print("shape of the prediction : ",output.shape)
		init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

		# print(init_pred)
		# print("shape of the prediction : ",init_pred.shape)
		
		# If the initial prediction is wrong, dont bother attacking, just move on
		if init_pred.item() != target.item():
			continue
		correctly_predicted_outputs_before_attack+=1	

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
		# Re-classify the perturbed image
		output = model(perturbed_data)

		# Check for success
		final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
		if final_pred.item() != target.item():
			incorrectly_predicted_outputs_after_attack += 1

			# Save some adv examples for visualization later
			if len(adv_examples) < 10:
				adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
				adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
	# Calculate final accuracy for this epsilon
	final_acc = incorrectly_predicted_outputs_after_attack/float(correctly_predicted_outputs_before_attack)
	print("FGSM Attack\nEpsilon: {}\nAttack Accuracy = (incorrectly predicted labels after attack) / (correctly predicted labels before attack) = {} / {} = {}\n\n".format(epsilon, incorrectly_predicted_outputs_after_attack, correctly_predicted_outputs_before_attack, final_acc))

	# Return the accuracy and an adversarial example
	return final_acc, adv_examples



# pgd attack code
def pgd_attack(model,image,label, epsilon, steps=10, alpha=2/255 ) :

	ori_image = image            
	for i in range(steps) :    
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

		adv_image = image + alpha*data_grad.sign()
		eta = torch.clamp(adv_image - ori_image, min=-epsilon, max=epsilon)
		image = torch.clamp(ori_image + eta, min=0, max=1).detach_()
	return image

def testPGD (model, device, test_loader, epsilon,steps=10) :
	# Accuracy counter
	incorrectly_predicted_outputs_after_attack = 0
	correctly_predicted_outputs_before_attack = 0
	adv_examples = []
	for data,target in tqdm(test_loader) :
		# Send the data and label to the device
		data, target = data.to(device), target.to(device)
		# Set requires_grad attribute of tensor. Important for Attack
		data.requires_grad = True

		# Forward pass the data through the model
		output = model(data)
		init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

		# If the initial prediction is wrong, dont bother attacking, just move on
		if init_pred.item() != target.item():
			continue
		correctly_predicted_outputs_before_attack+=1	


		perturbed_data = pgd_attack(model,data.to(device),target.to(device),epsilon,steps)

		# Re-classify the perturbed image
		output = model(perturbed_data)

		# Check for success
		final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
		if final_pred.item() != target.item():
			incorrectly_predicted_outputs_after_attack += 1
			# Save some adv examples for visualization later
			if len(adv_examples) < 10:
				adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
				adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

	# Calculate final accuracy for this epsilon
	final_acc = incorrectly_predicted_outputs_after_attack/float(correctly_predicted_outputs_before_attack)
	print("\nSteps: {}\nAttack Accuracy = (incorrectly predicted labels after attack) / (correctly predicted labels before attack) =  {} / {} = {}".format(steps, incorrectly_predicted_outputs_after_attack,correctly_predicted_outputs_before_attack, final_acc))

	PGD_step_wise_accuracy.append(final_acc)
	# Return the accuracy and an adversarial example
	return final_acc, adv_examples

#load model
model = create_model("fan_tiny_12_p16_224", img_size=224).to(device)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
#get validation accuracy without attack
accuracy = 0
for val_batch, val_label in tqdm(val_data_loader):
	val_batch = val_batch.to(device)
	val_label = val_label.to(device)
	pred = model(val_batch)
	# print("size of the prediction : ",pred.size)
	# print("shape of the prediction : ",pred.shape)
	# print("**********************************************")
	# print("size of the label : ",val_label.size)
	# print("shape of the label : ",val_label.shape)
	# print("**********************************************")
	prediction = torch.argmax(pred[0].cpu()).numpy()
	label = val_label.cpu().numpy()[0]
	if prediction == label :
		accuracy += 1

print(f"Model accuracy = {(accuracy/50000)*100}")

acc_FGSM, ex_FGSM = testFGSM(model, device, val_data_loader, 25/255)

for steps in [1,5] :
	acc_PGD, ex_PGD = testPGD(model, device, val_data_loader, 25/255,steps)


def show_adversarial_images(examples,labels, initial_name) :
	for j in range(len(examples)):
		plt.xticks([], [])
		plt.yticks([], [])
		orig,adv,ex = examples[j]
		plt.title("{} -> {}".format(labels[str(orig)],labels[str(adv)]))
		plt.imshow(ex[0].T)
		name = initial_name + time.strftime("%Y%m%d-%H%M%S") + '.png'
		plt.savefig('./' + name)
		plt.show()

show_adversarial_images(ex_FGSM,ground_labels, 'fsgm_')
show_adversarial_images(ex_PGD,ground_labels, 'pgd_')

#get validation accuracy after attack



