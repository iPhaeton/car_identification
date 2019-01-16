from PIL import Image
from io import BytesIO
import base64

def squash_width(img, width):
    w, h = img.size
    ratio = h / w
    return img.resize([width, int(width * ratio)])

def squash_height(img, height):
    w, h = img.size
    ratio = w / h
    return img.resize([int(height * ratio), height])

def get_image_of_size(path, size, preserve_ratio=False):
    width, height = size
    img = Image.open(path)

    if preserve_ratio == False:
        img = img.resize(size)
    else:
        w, h = img.size
        if w > width:
            img = squash_width(img, width)
            w, h = img.size
        if h > height:
            img = squash_height(img, height)

    img_buffer = BytesIO()
    img.save(img_buffer, format="JPEG")
    img_base64 = base64.b64encode(img_buffer.getvalue())
    return img_base64.decode("utf-8")
