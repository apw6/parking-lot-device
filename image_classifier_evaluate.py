import os
import numpy as np
import pandas as pd
from skimage import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pickle
import camera_config
from feature_extraction import get_image_features

camera = camera_config.camera

image_dir = "image data/"

# scale factors as the released data has rescaled images from 2592x1944 to 1000x750
x_scale = camera_config.x_scale
y_scale = camera_config.y_scale

image_resize_x = camera_config.image_resize_x
image_resize_y = camera_config.image_resize_y

weather_code_map = {"O": "OVERCAST",
                    "R": "RAINY",
                    "S": "SUNNY"}

with open("image data/LABELS/{}.txt".format(camera)) as label_txt:
    label_file_data = pd.read_csv(label_txt, delimiter=" ", header=None)


def create_image_set(path):
    labels = []
    patch_generator_metadata = []

    for weather in sorted(os.listdir(path)):

        weather_path = os.path.join(path, weather)
        for day in sorted(os.listdir(weather_path)):
            camera_path = os.path.join(weather_path, day, camera)
            for image in sorted(os.listdir(camera_path)):
                # load full image
                image_path = os.path.join(camera_path, image)

                # parse data from filename
                # weather code, date, time, camera, space id, occupied (0 = free)

                metadata = image.split("_")
                metadata[4] = metadata[4].split(".")[0]
                metadata.append(
                    label_file_data[label_file_data[0] == os.path.join(weather, day, camera, image)].iloc[0][1] == 1)

                full_image_path = os.path.join(camera_config.full_image_dir, weather_code_map[metadata[0]], metadata[1],
                                               camera,
                                               metadata[1] + "_" + "".join(metadata[2].split(".")) + ".jpg")

                patch_generator_metadata.append((int(metadata[4]), full_image_path))
                labels.append(metadata[5])

    def image_generator():
        for space_number, file_path in patch_generator_metadata:
            full_image_data = io.imread(file_path)
            patch_image_data = camera_config.get_image_patch(space_number, full_image_data)
            yield patch_image_data

    return image_generator, labels


def evaluate_model(model):
    # Testing

    print("creating test set generator")
    image_generator_function, test_labels = create_image_set(os.path.join(image_dir, "test"))
    print("Percent occupied examples in test: {} of {}".format(test_labels.count(True) / len(test_labels),
                                                               len(test_labels)))

    print("loading test set")
    features = [np.hstack(get_image_features(image)) for image in image_generator_function()]

    print("testing")
    predict_labels = model.predict(features)

    accuracy = accuracy_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    precision = precision_score(test_labels, predict_labels)

    print("Testing Confusion Mat:\n {}".format(confusion_matrix(test_labels, predict_labels)))

    result_string = "Accuracy: {}\n" \
                    "Recall: {}\n" \
                    "Precision: {}\n"
    result_string = result_string.format(accuracy, recall, precision)

    print(result_string)


if __name__ == "__main__":
    trained_model = classifier_model = pickle.load(open(camera_config.classifier_model_filename, 'rb'))
    evaluate_model(trained_model)
