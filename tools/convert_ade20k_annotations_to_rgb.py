from PIL import Image
import numpy as np
import os
from pathlib import Path
import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def create_colormap():
    hsv_colors = [(i / 150, 0.75, 0.75) for i in range(150)]
    random.Random(1337).shuffle(hsv_colors)
    rgb_colors = [mcolors.hsv_to_rgb(hsv) for hsv in hsv_colors]
    color_map = (np.array(rgb_colors) * 255).astype(np.uint8)

    return color_map
def convert_to_rgb(input, output, color_map):
    img = np.asarray(Image.open(input))
    H, W = img.shape
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
    for label in range(150):
        rgb_image[img == label] = color_map[label]
    image = Image.fromarray(rgb_image)

    image.save(output)


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "ADEChallengeData2016"
    for name in ["validation"]:
        annotation_dir = dataset_dir / "annotations_detectron2" / name
        output_dir = dataset_dir / "annotations_detectron2_rgb" / name
        output_dir.mkdir(parents=True, exist_ok=True)

        color_map = create_colormap()

        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            convert_to_rgb(file, output_file, color_map)
