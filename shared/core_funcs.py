import re
from pathlib import Path

import numpy as np
import skimage as sk


def import_image_stack(image_dir, roi=None):

    if isinstance(image_dir, str):
        image_dir = Path(image_dir)

    # Find all files and import them
    files = list(image_dir.glob("*.tif"))
    # print(files)

    # Sort the files by "_w#"
    def get_image_number(filepath):
        match = re.search(r"_w(\d+)", str(filepath))
        return int(match.group(1)) if match else 0
    
    files.sort(key=get_image_number)
    
    # print(files)

    images = sk.io.imread_collection(files)

    # NOTE: I'm not sure if it's the best idea to import all of the images at once, but will leave for now. Alternatively, we could just return the list of Paths.
    image_stack = sk.io.concatenate_images(images)

    if roi:
        image_stack = image_stack[:, 
                       roi[1]:(roi[1] + roi[3]), 
                       roi[0]:(roi[0] + roi[2])]

    return image_stack

