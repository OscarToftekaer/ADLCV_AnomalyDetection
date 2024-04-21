import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import seaborn as sns
from typing import Optional

sns.set_theme(style='whitegrid')

def plotROC(y_true, y_scores, grouping=None, save_plot:Optional[str]=None):
    """
    Plot the ROC curve for the provided true labels and scores.
    If grouping is provided, plot separate ROC curves for each group.

    Parameters:
    - y_true: Array-like, true binary labels.
    - y_scores: Array-like, target scores, probabilities of the positive class.
    - grouping: Optional, Array-like, grouping labels for each entry in y_true and y_scores.
    """
    if grouping is None:
        # Plot ROC curve for the entire dataset
        fpr, tpr, _ = roc_curve(y_true, y_scores)
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
        if save_plot is not None:
            plt.savefig(save_plot, transparent = True, dpi = 400)
        else:
            plt.show()

    else:
        # Plot ROC curve for each group
        df = pd.DataFrame({
            'y_true': y_true,
            'y_scores': y_scores,
            'grouping': grouping
        })

        plt.figure()
        for name, group in df.groupby('grouping'):
            fpr, tpr, _ = roc_curve(group['y_true'], group['y_scores'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label='ROC curve of %s (area = %0.2f)' % (name, roc_auc))

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic by Group')
        plt.legend(loc="lower right")
        
        if save_plot is not None:
            plt.savefig(save_plot, transparent = True, dpi = 400)
        else:
            plt.show()