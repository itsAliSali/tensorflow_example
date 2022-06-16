import random 
import numpy as np
import matplotlib.pyplot as plt

from mnist import MNIST


def plot(imgs, lbl, pre, w=12, h=5):
    
    _, ax = plt.subplots(h, w)

    for i in range(w*h):
        idx = random.randint(0, len(imgs))

        ax[i//w, i%w].imshow(imgs[idx], cmap='gray')
        ax[i//w, i%w].title.set_text(f'{lbl[idx]}/{pre[idx]}')
        ax[i//w, i%w].axis('off')

    plt.show()


if __name__ == "__main__":
    
    # loading dataset:
    data = MNIST("./data/")
    train_img, train_lbl = data.load_training()

    # type(), len(), .shape(), np.unique()

    # exit()
    # reshaping training data:
    train_img = np.array(train_img).reshape(60000, 28, 28)
    train_lbl = np.array(train_lbl)

    #normalizing 
    train_img = train_img / 255.0

    # plotting
    plot(train_img, train_lbl, train_lbl, 11, 5)
