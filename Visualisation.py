import matplotlib.pyplot as plt
import numpy as np


class Visualisation:

    def img_plot(self, title_text, x, y):
        fig = plt.figure(figsize=(8, 4))
        fig.suptitle(title_text)
        for i in range(10):
            ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(x[i]), cmap='gray')
            ax.set_title(label=y[i])
