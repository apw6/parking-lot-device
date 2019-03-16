import os
import pickle
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import mqtt_config
import camera_config
import json
import time
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from feature_extraction import get_image_features
from concurrent.futures import ThreadPoolExecutor
from functools import partial

parking_space_number_list = [601, 602, 603]
space_number_to_id_map = {
    601: 1,
    602: 2,
    603: 3,
}
classifier_model = pickle.load(open(camera_config.classifier_model_filename, 'rb'))

roi_data = camera_config.roi_data
x_scale = camera_config.x_scale
y_scale = camera_config.y_scale
image_resize_x = camera_config.image_resize_x
image_resize_y = camera_config.image_resize_y

executor = ThreadPoolExecutor(max_workers=5)
def detect_space_occupancy_in_image(image, classifier_model):
    # patches = executor.map(partial(camera_config.get_image_patch, image=image), parking_space_number_list)
    patches = [camera_config.get_image_patch(space_number, image=image) for space_number in parking_space_number_list]
    # plt.imshow(patches[2])
    # plt.show()

    # features = list(executor.map(lambda x: np.hstack(get_image_features(x)), patches))
    features = [np.hstack(get_image_features(patch)) for patch in patches]
    predictions = [bool(prediction) for prediction in classifier_model.predict(features)]
    wrapped_predictions = [{"number": number, "occupied": prediction} for number, prediction in
                           zip(parking_space_number_list, predictions)]
    return wrapped_predictions


# MQTT Setup
thing_name = mqtt_config.thing_name
root_ca_path = mqtt_config.root_ca_path
certificate_path = mqtt_config.certificate_path
private_key_path = mqtt_config.private_key_path
cloud_endpoint = mqtt_config.cloud_endpoint

myMQTTClient = AWSIoTMQTTClient(thing_name)
myMQTTClient.configureEndpoint(cloud_endpoint, 8883)
myMQTTClient.configureCredentials(root_ca_path,
                                  private_key_path,
                                  certificate_path)
myMQTTClient.configureOfflinePublishQueueing(-1)
myMQTTClient.configureDrainingFrequency(2)
myMQTTClient.configureConnectDisconnectTimeout(7)
myMQTTClient.configureMQTTOperationTimeout(5)

myMQTTClient.connect()

# Image Input Setup
image_path = os.path.join(camera_config.full_image_dir, "demo")
camera = camera_config.camera


def image_path_generator():
    images = [os.path.join(image_path, day, camera, image) for day in sorted(os.listdir(image_path)) for image in
              sorted(os.listdir(os.path.join(image_path, day, camera)))]
    index = 0
    while index < len(images):
        yield images[index]
        index = (index + 1) % len(images)


image_path_gen = image_path_generator()
while True:
    start_time = time.time()

    image_path = image_path_gen.__next__()
    image_data = io.imread(image_path)

    occupied = detect_space_occupancy_in_image(image_data, classifier_model)
    for space in occupied:
        space_id = space_number_to_id_map.get(space["number"])
        if not space_id:
            continue
        message = {"occupied": space["occupied"]}
        print("Sending: {}".format(message))
        myMQTTClient.publish("iotproject/parkingspace/{}".format(space_id),
                             json.dumps(message), 0)
    finish_time = time.time()
    print("Time {}".format(finish_time - start_time))
    plt.imshow(image_data)
    plt.show()
    time.sleep(1.5)
myMQTTClient.disconnect()
