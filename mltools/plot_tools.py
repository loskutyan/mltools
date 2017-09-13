import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, figsize=(15, 15)):
    classes_size = ['{} ({})'.format(x[0], x[1][0]) for x in zip(classes, cm.sum(axis=1)[:, np.newaxis])]
    if normalize:
        cm = np.nan_to_num(100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).astype('int')
        if title == 'Confusion matrix':
            title = "Normalized confusion matrix"
        print(title)
    else:
        if title == 'Confusion matrix':
            title = 'Confusion matrix, without normalization'
        print(title)
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes_size)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')