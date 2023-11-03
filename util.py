

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import random

device = torch.device("cpu")



def create_model(num_classes, model_type="single_label", dropout_rate=0.7):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_ftrs, num_classes),
        nn.Sigmoid() if model_type == "multi_label" else nn.Identity()
    )
    
    model = model.to(device)
    return model

class_names = ['Not_a_Valid_Image', 'Optimal', 'SubOptimal500', 'Wrong']
multi_label_class_names = ['Artefact', 'Incorrect_Gain', 'Incorrect_Position']
num_classes= 3

single_label_model = create_model(len(class_names), model_type="single_label", dropout_rate=0.7)
multi_label_model = create_model(len(multi_label_class_names), model_type="multi_label", dropout_rate=0.7)

single_label_weights = 'trained_multi_class_model.pth'
multi_label_weights = 'trained_multi_label_model.pth'

single_label_model.load_state_dict(torch.load(single_label_weights, map_location=torch.device('cpu')))
multi_label_model.load_state_dict(torch.load(multi_label_weights, map_location=torch.device('cpu')))

single_label_model.eval()
multi_label_model.eval()






def generate_feedback(predicted_classes):
    feedback = []
    for prediction in predicted_classes:
        feedback_for_prediction = []
        labels = prediction.split("__")  # Change this line to split by "__" instead of "_"
        for label in labels:
            if label == "Artefact":
                feedback_for_prediction.append('''You can avoid acoustic shadowing from the ribs by asking the subject to take a 
                                               deep inspiration, which will lower the diaphragm and hence the position of the
                                                kidney away from the ribcage, or by positioning the probe in between the ribs to 
                                               avoid the artefact. Fasting or gently applying pressure on the probe to displace 
                                               the gas away from the area may partially overcome ring-down artefacts from bowel gas.
                                                <a href="https://moodle.hku.hk/mod/page/view.php?id=2748898" target="_blank">Please Visit Moodle</a>''')
                
            elif label == "Incorrect_Gain":
                feedback_for_prediction.append('''The image is either too "bright" or too "dark". <a href="https://moodle.hku.hk/mod/page/view.php?id=2748898" target="_blank">Please Visit Moodle</a> or
                                                <a href="https://123sonography.com/blog/ultrasound-101-part-5-gain-and-time-gain-compensation#:~:text=What%20is%20gain%3F,much%20each%20echo%20is%20amplified." target="_blank">Visit Sonography123</a>''')
                
            elif label == "Incorrect_Position":
                feedback_for_prediction.append('''The kidney is not centrally placed or incompletely imaged. 
                                               Having the subject in decubitus position helps to get good access to 
                                               image the kidney (<a href="https://moodle.hku.hk/mod/page/view.php?id=2748898" target="_blank">Please Visit Moodle</a>) ''')
                
            elif label == "Optimal":
                feedback_for_prediction.append('''Well done/good work for obtaining optimal image quality of the kidney.''')
                
            elif label == "Wrong":
                feedback_for_prediction.append('''The image acquired is not of the kidney. <a href="https://moodle.hku.hk/mod/page/view.php?id=2748898" target="_blank">Please Visit Moodle</a>''')
            
            elif label == "Wrong":
                feedback_for_prediction.append('''The image is a not valid image. <a href="https://moodle.hku.hk/mod/page/view.php?id=2748898" target="_blank">Please Visit Moodle</a>''')

            else:
                feedback_for_prediction.append('''I am sorry, something went wrong''')
        feedback.append(feedback_for_prediction)
    return feedback
