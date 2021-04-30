import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

rng = np.random.default_rng()

class Visualisation:

    def img_plot(self, title_text, x, y):
        fig = plt.figure(figsize=(8, 4))
        fig.suptitle(title_text)
        sample = rng.choice(len(y), 10)
        for i, j in enumerate(sample):
            ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(x[j]), cmap='gray')
            ax.set_title(label=y[j])

    def cm_plot(self, y_test, y_preds, labels):
        cm = confusion_matrix(y_test, y_preds)

        df_cm = pd.DataFrame(cm, index=labels,
                             columns=labels)
        df_cm.index.name = "Actual"
        df_cm.columns.name = "Predicted"
        plt.figure(figsize=(8, 6))
        sn.heatmap(df_cm, cmap="Blues", annot=True, fmt='d')
        plt.show()
