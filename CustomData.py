import cv2
import numpy as np

rng = np.random.default_rng()


class CustomData:
    """class for data generation and preprocessing"""

    def generate_data(self, n):
        """generates array with images and list of corresponding labels

        :param n: sample size (int)
        :return: array containing image data and list containing labels
        """

        def rand_shape():
            def make_line():
                img = np.zeros((64, 64))
                coord1 = tuple(rng.integers(4, 60, size=2))
                coord2 = tuple(rng.integers(4, 60, size=2))
                cv2.line(img, coord1, coord2, color=(255, 255, 255), thickness=1)
                label = "line"
                return img, label

            def make_circle():
                img = np.zeros((64, 64))
                coord = tuple(rng.integers(24, 40, size=2))
                size = rng.integers(5, 20)
                cv2.circle(img, coord, size, color=(255, 255, 255), thickness=1)
                label = "circle"
                return img, label

            def make_rectangle():
                img = np.zeros((64, 64))
                coord_l = tuple(rng.integers(4, 30, size=2))
                coord_r = tuple(rng.integers(34, 60, size=2))
                cv2.rectangle(img, coord_l, coord_r, color=(255, 255, 255), thickness=1)
                label = "rectangle"
                return img, label

            def make_triangle():
                img = np.zeros((64, 64))
                pts = np.array(rng.integers(4, 60, size=6))
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=1)
                label = "triangle"
                return img, label

            def make_chair():
                img = np.zeros((64, 64))
                center = rng.integers(24, 40)
                size = rng.integers(5, 20)
                line2 = tuple([center - size, center - size, center + size, center + size])
                line1 = tuple([center - size, center + size, center + size, center - size])
                lines = np.array([line1, line2]).reshape(-1, 2)
                cv2.polylines(img, [lines], isClosed=False, color=(255, 255, 255), thickness=1)
                label = "chair"
                return img, label

            funs = [make_line, make_circle, make_rectangle, make_triangle, make_chair]
            return rng.choice(funs)()

        x = []
        y = []
        for _ in range(n):
            img, label = rand_shape()
            x.append(img)
            y.append(label)
        x = np.asarray(x)
        return x, y

    def preprocessing(self, x, y):
        """preprocesses input data.
        Includes reshaping, normalizing of pixel values and turning string labels to integers

        :param x: image array
        :param y: list of labels
        :return: reshaped and normalized img array, list of corresponding labels (int), list of unique original labels
        """
        # reshaping image data
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))

        # normalizing pixel values
        x = x / 255.0

        # string labels to integers
        if isinstance(y[0], str):
            labels, y = np.unique(y, return_inverse=True)
        else:
            labels = np.unique(y)

        return x, y, labels
