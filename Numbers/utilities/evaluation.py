from os.path import dirname, join
import matplotlib.pyplot as plt

def generate_number_images(images, suptitle, filename):
    figure, axes = plt.subplots(2, 5, figsize=(10,4))
    for c, ax in enumerate(axes.flat):
        if images[c] is not None:
            ax.imshow(images[c], cmap='gray')
        ax.set_title(str(c))
        ax.axis('off')

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(join(dirname(dirname(__file__)), 'results', filename))