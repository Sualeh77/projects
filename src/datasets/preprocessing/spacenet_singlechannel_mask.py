import os, json, shutil
import georaster
from skimage.draw import polygon
import numpy as np
from tqdm import tqdm
from PIL import Image

class SpaceNetDataConverter:
    """
        Directory should be in the following structure:

        root_dir/
            geojsons/ (filled with GeoJSON data for each building) each file should be in .geojson data
                buildings_AOI_<AOI_ID>_<LOCATION>_img<ID>.geojson
            raw_tif/ (filled with raw RBG Pan Sharpened Tif Images)
                RBG-PanSharpen_AOI_<AOI_ID>_<LOCATION>_img<ID>.tif

        save_dir/ (required)
            images/ (optional)
            masks/ (optional)
    """
    
    def __init__(self, root_dir, save_dir, aoi_id, location, output_format="npy"):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.aoi_id = aoi_id
        self.location = location
        self.output_format = output_format

        self.image_prefix = "RGB-PanSharpen_AOI_{}_{}_img".format(self.aoi_id, self.location)
        self.mask_prefix = "buildings_AOI_{}_{}_img".format(self.aoi_id, self.location)

        geojson_files = os.listdir(os.path.join(root_dir, 'geojson/buildings'))
        self.ids = [g[g.index('img') + 3: g.index('.')] for g in geojson_files]

        if self.output_format not in os.listdir(save_dir):
            os.mkdir(f"{self.save_dir}/{self.output_format}")
            os.mkdir(f"{self.save_dir}/{self.output_format}/masks")
            os.mkdir(f"{self.save_dir}/{self.output_format}/images")
        
    def convertAllToInput(self):
        self.convertToInput(0, len(self.ids))
        
    def convertToInput(self, start, end):
        num_blank = 0
        pbar = tqdm(self.ids)

        downsample = 1

        pbar.set_description("{} samples are blank".format(num_blank))
        for img_id in pbar:
            tiff_file = os.path.join(self.root_dir, "RGB-PanSharpen", self.image_prefix + img_id + ".tif")
            tiff = georaster.MultiBandRaster(tiff_file, downsampl=int(downsample))

            geojson_file = os.path.join(self.root_dir, "geojson/buildings", self.mask_prefix + img_id + ".geojson")
            with open(geojson_file) as gf:
                geojson = json.load(gf)

            # SpaceNet satellite imagery is stored as 11-bit data (values 0-2048)
            # Dividing by 2048 normalizes to 0-1 range before converting to 8-bit (0-255) RGB later
            image = tiff.r / 2048.0
            mask = self.geoJsonToMask(geojson, tiff)
            if mask is None:
                num_blank += 1
                pbar.set_description("{} samples are blank".format(num_blank))
                continue

            mask = mask.astype(np.uint8)
            image = (image * 255).astype(np.uint8)
            
            if self.output_format == "npy":
                # save images
                np.save(os.path.join(self.save_dir, f"{self.output_format}/images", img_id), image)
                # save masks
                np.save(os.path.join(self.save_dir, f"{self.output_format}/masks", img_id + "_mask"), mask)
            
            elif self.output_format == "png":
                # Convert and save images as PNG
                image_path = os.path.join(self.save_dir, f"{self.output_format}/images", img_id + ".png")
                Image.fromarray(image).save(image_path)

                # Scale mask to 0-255 for visibility
                mask_scaled = (mask * 255).astype(np.uint8)
                # Convert and save masks as PNG
                mask_path = os.path.join(self.save_dir, f"{self.output_format}/masks", img_id + "_mask.png")
                Image.fromarray(mask_scaled).save(mask_path)

        print("Finished!")

    def geoJsonToMask(self, geojson, tiff):
        polyMasks = np.zeros((650,650))
        for i,bldg in enumerate(geojson['features']):
            feature_type = bldg['geometry']['type']
            if 'Polygon' not in feature_type:
                continue
                
            polygons = [bldg['geometry']['coordinates']] if feature_type == "Polygon" else bldg['geometry']['coordinates']

            for mask in polygons:
                rasteredPolygon = np.array(mask[0])
                xs, ys = tiff.coord_to_px(rasteredPolygon[:,0], rasteredPolygon[:,1], latlon=True)

                cc, rr = polygon(xs, ys)
                # Clip indices to ensure they are within bounds
                rr = np.clip(rr, 0, polyMasks.shape[0] - 1)
                cc = np.clip(cc, 0, polyMasks.shape[1] - 1)
                polyMasks[rr, cc] = 1


        if len(geojson['features']) > 0:
            assert np.max(polyMasks) == 1 and np.min(polyMasks) == 0
            if np.sum(polyMasks) <= 5:
                return None
        else:
            return None

        return polyMasks

# Please use this function to split SpaceNet Train into Train/Val      
def train_val_split(root_dir, save_dir, train_percent):
    all_images = list(os.listdir(os.path.join(root_dir, "images")))
    np.random.shuffle(all_images)

    num_train = int(len(all_images) * train_percent)
    val_images = all_images[num_train:]

    for img in tqdm(val_images):
        shutil.move(os.path.join(root_dir, "images", img), os.path.join(save_dir, "images"))
        shutil.move(os.path.join(root_dir, "masks", img.replace(".npy", "_mask.npy")), os.path.join(save_dir, "masks"))

if __name__ == "__main__":
    """

    Example Usage:
        converter = SpaceNetDataConverter('/data/SpaceNet/AOI_2_Vegas_Train', '/data/SpaceNet/Vegas/train', 2, "Vegas")
        converter.convertAllToInput()

        train_val_split("/data/SpaceNet/Vegas/train", "/data/SpaceNet/Vegas/val", 0.8)
    """

        