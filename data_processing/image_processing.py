from PIL import Image,ImageOps
from pillow_heif import register_heif_opener
import os
import numpy as np
import cv2


register_heif_opener()

os.makedirs('data_set', exist_ok=True)
os.makedirs('falty_data', exist_ok=True)

root = '/Volumes/MINISHAN/Invoice Data'
threshold = 100
new_size = (1536,1536)
extension = '.jpg'


def is_blurry(image : np.ndarray) -> bool:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    laplacian_var = cv2.Laplacian(image,cv2.CV_64F).var()
    if laplacian_var < threshold: return True
    else: return False


i = 1
for dirpath,dirs,files in os.walk(os.path.join(root)):
    for file in files:
        if file.startswith('._'):
            continue
        image = Image.open(os.path.join(root,file))
        exif_data = image.getexif()
        if exif_data.get(274,1) != 1:
            image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')
        image.thumbnail(new_size,Image.Resampling.LANCZOS)

        
        blur = is_blurry(np.asarray(image))
        if blur: image.save(f'falty_data/{i}{extension}')
        else: image.save(f'data_set/{i}{extension}')
        i+=1



print('done')

