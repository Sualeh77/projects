import os
import json
import numpy as np
import solaris as sol
import rasterio
from rasterio.plot import reshape_as_image
from tqdm import tqdm
import argparse


class SpaceNetMultiChannelMaskConverter:
    def __init__(self, root_dir, save_dir, aoi_id, location):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.aoi_id = aoi_id
        self.location = location

        self.image_prefix = "RGB-PanSharpen_AOI_{}_{}_img".format(self.aoi_id, self.location)
        self.mask_prefix = "buildings_AOI_{}_{}_img".format(self.aoi_id, self.location)

        geojson_files = os.listdir(os.path.join(root_dir, 'geojson/buildings'))
        self.ids = [g[g.index('img') + 3: g.index('.')] for g in geojson_files]

        if 'masks' not in os.listdir(save_dir):
            os.mkdir(f"{save_dir}/masks")
            
        if 'images' not in os.listdir(save_dir):
            os.mkdir(f"{save_dir}/images")

    def convertAllToMultiChannelMask(self):
        for img_id in tqdm(self.ids):
            self.createMultiChannelMask(img_id)

    def createMultiChannelMask(self, img_id):
        tiff_file = os.path.join(self.root_dir, "RGB-PanSharpen", self.image_prefix + img_id + ".tif")
        geojson_file = os.path.join(self.root_dir, "geojson/buildings", self.mask_prefix + img_id + ".geojson")

        # Load the TIFF image using rasterio
        with rasterio.open(tiff_file) as src:
            image = src.read()
            image = reshape_as_image(image)  # Reshape to (height, width, channels)
            image = image / 2048.0  # Normalize the image to 0-1 range, 2048 is the max value for 11-bit data 
            image = (image * 255).astype(np.uint8)  # Scale to 0-255

        # Create multi-channel mask using solaris
        fbc_mask = sol.vector.mask.df_to_px_mask(
            df=geojson_file,
            channels=['footprint', 'boundary', 'contact'],
            reference_im=tiff_file,
            boundary_width=5,
            contact_spacing=10,
            meters=True
        )

        # Check if the mask is valid
        if fbc_mask is None or np.sum(fbc_mask) == 0:
            print(f"Skipping {img_id} as it has no valid mask.")
            return

        # Convert mask to uint8
        fbc_mask = fbc_mask.astype(np.uint8)

        # Save the image and multi-channel mask
        np.save(os.path.join(self.save_dir, "images", img_id), image)
        np.save(os.path.join(self.save_dir, "masks", img_id + "_multi_channel_mask"), fbc_mask)


def main():
    parser = argparse.ArgumentParser(description="Convert SpaceNet data to multi-channel masks using Solaris.")
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the root directory containing the data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to the directory where processed data will be saved.')
    parser.add_argument('--aoi_id', type=int, required=True, help='Area of Interest (AOI) ID.')
    parser.add_argument('--location', type=str, required=True, help='Location name.')

    args = parser.parse_args()

    converter = SpaceNetMultiChannelMaskConverter(
        args.root_dir,
        args.save_dir,
        args.aoi_id,
        args.location
    )
    converter.convertAllToMultiChannelMask()

if __name__ == "__main__":
    main()