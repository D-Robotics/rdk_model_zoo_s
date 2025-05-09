"""
 Copyright (c) 2021-2024 D-Robotics Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

# Import necessary packages
import numpy as np
import cv2
import pyclipper
import matplotlib.pyplot as plt
import collections
from hobot_dnn import pyeasy_dnn
import libmodel_task
from PIL import Image, ImageDraw, ImageFont
from postprocess.rec_postprocess import CTCLabelDecode


# Function definitions
def load_image(img_path):
    """Load an image from a file path."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image file '{img_path}' not found.")
    return img


def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]


def get_bounding_boxes(dilated_polys, min_area):
    """Get bounding boxes from dilated polygons."""
    boxes_list = []
    for cnt in dilated_polys:
        if cv2.contourArea(cnt) < min_area:
            continue
        rect = cv2.minAreaRect(cnt)  # Calculate the minimum enclosing rectangle
        box = cv2.boxPoints(rect).astype(np.int_)
        boxes_list.append(box)
    return boxes_list


def dilate_contours(contours, ratio_prime):
    """Dilate contours using the ratio_prime."""
    dilated_polys = []
    for poly in contours:
        poly = poly[:, 0, :]  # Extract the vertex coordinates of the polygon
        arc_length = cv2.arcLength(poly, True)  # Calculate the perimeter of the polygon
        if arc_length == 0:
            continue
        D_prime = (cv2.contourArea(poly) * ratio_prime / arc_length)  # Calculate the expansion amount D_prime

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        dilated_poly = np.array(pco.Execute(D_prime))

        if dilated_poly.size == 0 or dilated_poly.dtype != np.int_ or len(dilated_poly) != 1:
            continue
        dilated_polys.append(dilated_poly)
    return dilated_polys


def resize_to_even(image):
    """Resize the image to ensure its width and height are even."""
    resized_image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    return resized_image


def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[: height * width] = y
    nv12[height * width:] = uv_packed
    return nv12



def draw_bbox(img, bboxes, color=(128, 240, 128), thickness=3):
    img_copy = img.copy()
    for bbox in bboxes:
        bbox = bbox.astype(int)
        cv2.polylines(img_copy, [bbox], isClosed=True, color=color, thickness=thickness)
    return img_copy


def crop_and_rotate_image(img, box):
    """Crop the image using the bounding box coordinates."""
    rect = cv2.minAreaRect(box)
    box = cv2.boxPoints(rect).astype(np.intp)
    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = rect[2]

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]],
                       dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))

    if angle >= 45:
        rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    else:
        rotated = warped

    print("width:", rotated.shape[1], "height:", rotated.shape[0])

    return rotated


def resize_norm_image(img, input_size):
    image_resized = cv2.resize(img, dsize=(input_size[1], input_size[0]))
    image_resized = (image_resized / 255.0).astype(np.float32)
    input_image = np.zeros((image_resized.shape[0], image_resized.shape[1], 3), dtype=np.float32)
    input_image[:image_resized.shape[0], :image_resized.shape[1], :] = image_resized
    input_image = image_resized[:, :, [2, 1, 0]]  # bgr->rgb
    input_image = input_image[None].transpose(0, 3, 1, 2)  # NHWC -> HCHW

    return input_image


def rec_predict(input_image, model, output_size, postprocess_op):
    input_data = [input_image]
    res_outputs = model.ModelInfer(input_data)

    preds = np.array(res_outputs, dtype=np.float32).reshape(1, *output_size)
    results = postprocess_op(preds)
    return results[0][0]


def draw_text_on_image(img, texts, boxes, font_path, font_size=20, color=(0, 0, 0)):
    """Draw text on an image."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    for text, box in zip(texts, boxes):
        center_x = int((box[0][0] + box[2][0]) / 2)
        center_y = int((box[0][1] + box[2][1]) / 2)

        text_width, text_height = draw.textsize(text, font=font)

        text_x = center_x - text_width // 2
        text_y = center_y - text_height // 2

        draw.text((text_x, text_y), text, font=font, fill=color)

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img


# Main process
model_path = 'cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm'
img_path = "gt_2322.jpg"
origin_img = load_image(img_path)

detection_model = pyeasy_dnn.load(model_path)
threshold = 0.5
ratio_prime = 2.7
input_size = (640, 640)

h, w = get_hw(detection_model[0].inputs[0].properties)
print(f"Model input size: {h}x{w}")


reshape_img = resize_to_even(origin_img)

nv12_image = bgr2nv12_opencv(reshape_img)

img_shape = origin_img.shape[:2]

outputs = detection_model[0].forward(nv12_image)
preds = np.array(outputs[0].buffer, dtype=np.float32).reshape(1, *input_size)
preds = np.where(preds > threshold, 255, 0).astype(np.uint8).squeeze()
preds = cv2.resize(preds, (img_shape[1], img_shape[0]))

contours, _ = cv2.findContours(preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
dilated_polys = dilate_contours(contours, ratio_prime)
boxes_list = get_bounding_boxes(dilated_polys, 100)

img_boxes = draw_bbox(origin_img, boxes_list)


cropped_images = []
for i, box in enumerate(boxes_list):
    cropped_img = crop_and_rotate_image(origin_img, box)
    cropped_images.append(cropped_img)

output_size = (40, 6625)
input_size = (48, 320)
recognition_model_path = 'cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm'
rec_model = libmodel_task.ModelTask()
rec_model.ModelInit(recognition_model_path)
postprocess_op = CTCLabelDecode("ppocr_keys_v1.txt")

recognized_texts = []
for i, img in enumerate(cropped_images):
    img = resize_norm_image(img, input_size)
    print(f"Box {i + 1}:")
    sim_pred = rec_predict(img, rec_model, output_size, postprocess_op)
    recognized_texts.append(sim_pred)
    print(f"Prediction: {sim_pred} \n")

font_path = "/usr/share/fonts/truetype/fangsong.ttf"
white_image = np.ones(origin_img.shape, dtype=np.uint8) * 255
img_with_text = draw_text_on_image(white_image, recognized_texts, boxes_list, font_path, font_size=35, color=(0, 0, 255))
combined_image = np.hstack((img_boxes, img_with_text))
cv2.imwrite("result.jpg", combined_image) 