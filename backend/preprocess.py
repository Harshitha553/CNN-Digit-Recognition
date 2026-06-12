import cv2

def preprocess_image(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise Exception(f"Unable to read image: {path}")

    img = cv2.resize(img, (28, 28))

    img = 255 - img

    img = img / 255.0

    img = img.reshape(1, 28, 28, 1)

    return img