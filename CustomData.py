import numpy as np
from PIL import Image, ImageDraw

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
                img = Image.new(mode="L", size=(64, 64))
                coord1 = tuple(rng.integers(4, 60, size=2))
                coord2 = tuple(rng.integers(4, 60, size=2))
                d = ImageDraw.Draw(img)
                d.line([coord1, coord2], fill="white")
                label = "line"
                return np.array(img), label

            def make_ellipse():
                img = Image.new(mode="L", size=(64, 64))
                coord1 = tuple(rng.integers(4, 30, size=2))
                coord2 = tuple(rng.integers(34, 60, size=2))
                d = ImageDraw.Draw(img)
                d.ellipse([coord1, coord2], fill=None, outline="white")
                label = "ellipse"
                return np.array(img), label

            def make_rectangle():
                img = Image.new(mode="L", size=(64, 64))
                coord_l = tuple(rng.integers(4, 30, size=2))
                coord_r = tuple(rng.integers(34, 60, size=2))
                d = ImageDraw.Draw(img)
                d.rectangle([coord_l, coord_r], fill=False, outline="white")
                label = "rectangle"
                return np.array(img), label

            def make_triangle():
                img = Image.new(mode="L", size=(64, 64))
                pts = list(rng.integers(4, 60, size=6))
                d = ImageDraw.Draw(img)
                d.polygon(pts, fill=None, outline="white")
                label = "triangle"
                return np.array(img), label

            def make_hourglass():
                img = Image.new(mode="L", size=(64, 64))
                center = rng.integers(24, 40)
                size = rng.integers(5, 20)
                lines = [(center - size, center - size), (center + size, center + size),
                         (center - size, center + size), (center + size, center - size)]
                d = ImageDraw.Draw(img)
                d.polygon(lines, fill=None, outline="white")
                label = "hourglass"
                return np.array(img), label

            funs = [make_line, make_ellipse, make_triangle, make_rectangle, make_hourglass]
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
