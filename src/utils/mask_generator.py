import random
import numpy as np
import skimage
from PIL import Image
from polygenerator import random_polygon, random_star_shaped_polygon, random_convex_polygon

def get_input_image(image: Image.Image, min_polygon_bbox_size: int = 50) -> dict:
    width, height = image.size

    while True:
        bbox_x1 = random.randint(0, width - min_polygon_bbox_size)
        bbox_y1 = random.randint(0, height - min_polygon_bbox_size)
        bbox_x2 = random.randint(bbox_x1, width)
        bbox_y2 = random.randint(bbox_y1, height)

        if (bbox_x2 - bbox_x1) < min_polygon_bbox_size or (bbox_y2 - bbox_y1) < min_polygon_bbox_size:
            continue

        mask_width = bbox_x2 - bbox_x1
        mask_height = bbox_y2 - bbox_y1

        num_points = random.randint(3, 20)
        polygon_func = random.choice([random_polygon, random_star_shaped_polygon, random_convex_polygon])

        polygon = polygon_func(num_points=num_points)
        polygon = [(round(r * mask_width), round(c * mask_height)) for r, c in polygon]
        polygon_mask = skimage.draw.polygon2mask((mask_width, mask_height), polygon)

        if np.sum(polygon_mask) > (min_polygon_bbox_size // 2) ** 2:
            break

    full_image_mask = np.zeros((width, height), dtype=np.uint8)
    full_image_mask[bbox_x1:bbox_x2, bbox_y1:bbox_y2] = polygon_mask

    image_gray = image.convert("L")
    image_gray_array = np.array(image_gray)
    random_color = random.randint(0, 255)
    image_gray_array[full_image_mask == 1] = random_color
    image_gray_masked = Image.fromarray(image_gray_array)

    return {
        "image_gt": image,
        "mask": full_image_mask,
        "image_gray": image_gray,
        "image_gray_masked": image_gray_masked,
    }
