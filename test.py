






import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from sklearn.metrics import precision_recall_fscore_support
from util import (class_names, num_classes, single_label_model, 
                  single_label_weights, multi_label_model, multi_label_weights, device, multi_label_class_names)


# Define folder paths
input_folder = 'testing_imgs'
output_folder = 'PREDS'

# Data preprocessing
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])



# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert('RGB')

        # Preprocess the image
        input_tensor = data_transform(img)
        input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
        input_batch = input_batch.to(device)

        # Make predictions with the single-label classifier
        with torch.no_grad():
            preds1 = single_label_model(input_batch)
            class1 = torch.argmax(preds1).item()

        # If the image is SubOptimal500, use the multi-label classifier
        if class_names[class1] == 'SubOptimal500':
            with torch.no_grad():
                preds2 = multi_label_model(input_batch)
                class2 = (preds2 > 0.5).squeeze().cpu().numpy().astype(int)

            # Create a prefix for the new filename
            prefix = '_'.join([multi_label_class_names[i] for i, c in enumerate(class2) if c == 1]) + '_'

            # Save the image with the new prefix
            output_path = os.path.join(output_folder, prefix + filename)
            img.save(output_path)

        else:
            # Save the image with the original name in the output folder
            output_path = os.path.join(output_folder, class_names[class1] + '_' + filename)
            img.save(output_path)

print("Prediction and renaming completed.")
