import cv2
import numpy as np
import secrets


class CustomData:

    def generate_data(self, n):
        def rand_shape():
            def make_line():
                img = np.zeros((64, 64))
                coord1 = tuple(np.random.randint(4, 60, size=2))
                coord2 = tuple(np.random.randint(4, 60, size=2))
                cv2.line(img, coord1, coord2, color=(255, 255, 255), thickness=1)
                label = "line"
                return img, label

            def make_circle():
                img = np.zeros((64, 64))
                coord = tuple(np.random.randint(24, 40, size=2))
                size = np.random.randint(5, 20)
                cv2.circle(img, coord, size, color=(255, 255, 255), thickness=1)
                label = "circle"
                return img, label

            def make_rectangle():
                img = np.zeros((64, 64))
                coord_l = tuple(np.random.randint(4, 30, size=2))
                coord_r = tuple(np.random.randint(34, 60, size=2))
                cv2.rectangle(img, coord_l, coord_r, color=(255, 255, 255), thickness=1)
                label = "rectangle"
                return img, label

            def make_triangle():
                img = np.zeros((64, 64))
                pts = np.array(np.random.randint(4, 60, size=6))
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=1)
                label = "triangle"
                return img, label

            def make_chair():
                img = np.zeros((64, 64))
                center = np.random.randint(24, 40)
                size = np.random.randint(5, 20)
                line2 = tuple([center-size, center-size, center+size, center+size])
                line1 = tuple([center-size, center+size, center+size, center-size])
                lines = np.array([line1, line2]).reshape(-1, 2)
                cv2.polylines(img, [lines], isClosed=False, color=(255, 255, 255), thickness=1)
                label = "chair"
                return img, label

            funs = [make_line(), make_circle(), make_rectangle(), make_triangle(), make_chair()]
            return secrets.choice(funs)

        x = []
        y = []
        for _ in range(n):
            img, label = rand_shape()
            x.append(img)
            y.append(label)
        x = np.asarray(x)
        return x, y

    def preprocessing(self, x, y):
        # reshaping the training and testing data
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))

        # normalizing pixel values
        x = x / 255.0

        # str labels to integers
        if isinstance(y[0], str):
            labels, y = np.unique(y, return_inverse=True)
        else:
            labels = np.unique(y)

        return x, y, labels
