import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix,ConfusionMatrixDisplay
from clf_dataset import CLF_Dataset
from sample_dataset import samples_Dataset
from classifiermodel import ResNet18
from helpers import create_result_folders
from torch.utils.data import DataLoader
import seaborn as sns
from torchvision.transforms import transforms

# Importing metrics and plot function
from metrics import accuracy, precision, recall, f1
from plots import plotROC

# Set seaborn style
sns.set_theme(style='whitegrid')

# Constants
DATA_DIR = '/dtu/datasets1/ashery-chexpert/data/inference_split'
IMG_SIZE = 128
BATCH_SIZE = 32

def load_data(img_size, batch_size,ddim=False):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize(size=(img_size, img_size))
    ])

    # Initialize dataset
    if ddim == False:
        DATA_DIR = '/dtu/datasets1/ashery-chexpert/data/inference_split'
        dataset = CLF_Dataset(transform=transform, data_dir=DATA_DIR,inference=True)
    else:
        DATA_DIR = '/work3/s194632/samples_ddim'
        dataset = samples_Dataset(transform=transform, data_dir=DATA_DIR)
    # Define data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader

def compute_scores(model, data_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    return all_predictions, all_labels

if __name__ == '__main__':
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(pretrained=True, num_classes=2)  # Assuming 2 classes (healthy, ill)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load('/zhome/76/b/147012/ADLCV_AnomalyDetection/models/clf/small_batch_diagnostic_best_model_checkpoint.pth'
                                     , map_location=device))
    # Load data
    data_loader = load_data(IMG_SIZE, BATCH_SIZE,ddim=False)

    # Compute scores
    predictions, labels = compute_scores(model, data_loader)

    print('scores:')
    print(predictions)
    print(sum(predictions))
    print('labels')
    print(labels)
    print('accuracy')
    print(accuracy(predictions,labels))
    cm = confusion_matrix(y_true=labels,y_pred=predictions)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['healthy','ill'])
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('./results/clf/ROC_diagnostic.png', transparent = True, dpi = 400)


    #plt.figure()
    disp.plot()
    plt.savefig('./results/clf/confusion_matrix.png', transparent = True, dpi = 400)
