import pandas as pd
import os

camera = "camera2"

image_dir = "image data"
full_image_dir = os.path.join(image_dir, "FULL_IMAGE_1000x750")

# scale factors as the released data has rescaled images from 2592x1944 to 1000x750
x_scale = 1000 / 2592
y_scale = 750 / 1944

image_resize_x = 150
image_resize_y = 150

gray_scale_threshold = 0.2

classifier_model_filename = camera + "_pickled_model"

with open("image data/FULL_IMAGE_1000x750/{}.csv".format(camera)) as roi_csv:
    roi_data = pd.read_csv(roi_csv)


def get_image_patch(space_number, image):
    try:
        row = roi_data[roi_data.SlotId == space_number].iloc[0]
    except IndexError as e:
        return None

    def convert_to_scale():
        return (int(row.X * x_scale),
                int(row.Y * y_scale),
                int(row.W * x_scale),
                int(row.H * y_scale))

    x, y, width, height = convert_to_scale()
    return image[y:y + height, x:x + width]
