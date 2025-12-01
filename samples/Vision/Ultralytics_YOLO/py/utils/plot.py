import cv2
from .coco_names import coco_names

rdk_colors = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
    (168, 153, 44),
    (255, 194, 0),
    (147, 69, 52),
    (255, 115, 100),
    (236, 24, 0),
    (255, 56, 132),
    (133, 0, 82),
    (255, 56, 203),
    (200, 149, 255),
    (199, 55, 255),
]


def draw_detection(img, bbox, score, class_id) -> None:
    """
    Draws a detection bounding box and label on the image.
    Parameters:
        img (np.array): The input image.
        bbox (tuple[int, int, int, int]): A tuple containing the bounding box coordinates (x1, y1, x2, y2).
        score (float): The detection score of the object.
        class_id (int): The class ID of the detected object.
    """
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id % 20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img,
        (label_x, label_y - label_height),
        (label_x + label_width, label_y + label_height),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        img,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
