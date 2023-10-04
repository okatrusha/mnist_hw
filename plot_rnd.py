
import numpy as np
import matplotlib.pyplot as plt

def rplt(X,y):

    np.random.seed(42)

    #Creating a list of 25 random integer numbers from the training set
    random_indices = np.random.randint(0, X.shape[0], 12)

    #Adding the images and label at randomly generated index to the lists
    images=[]
    labels=[]
    for i in random_indices:
        images.append(X[i].reshape((28,28,1)))
        labels.append(y[i])

    # Plot the images
    plt.subplots(figsize = (8, 8))
    for i, image in enumerate(images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(labels[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()