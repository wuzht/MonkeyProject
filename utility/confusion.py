import itertools
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import io


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    # Normalize the confusion matrix.
    cm_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=4)

    figure = plt.figure(figsize=(10, 10))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm_norm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm_norm[i, j] > threshold else "black"
        plt.text(j, i, '{}{}'.format(
            int(cm[i, j]), '\n({:.2f}%)'.format(100 * cm_norm[i, j]) if i == j else ''
        ), horizontalalignment="center", verticalalignment='center', color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


if __name__ == "__main__":
    cm = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])
    print(cm)
    figure = plot_confusion_matrix(cm, np.array(['0', '1', '2']))