from pathlib import Path
from PIL import Image
import numpy as np

class OriginalImage():
    def __init__(self,directory=Path('..','..','13','medical_data','grand_challenge','us_data_2D','training_set')):
        self.directory = directory
        self.image_extension='.png'
        self.mask_suffix='_Annotation'

    def set_image_name(self,file_name):
        self.file_name = self.directory.joinpath(file_name)
        self.mask_file_name = self.file_name.parent.joinpath(self.file_name.stem + self.mask_suffix + self.image_extension)
        #print(self.file_name)
        #print(self.mask_file_name)

    def get_image_width(self):
        return self.image_width

    def get_image_height(self):
        return self.image_height

    def read_images(self):
        self.raw_image = Image.open(self.file_name)
        raw_size = np.size(self.raw_image)
        self.mask_image = Image.open(self.mask_file_name)
        mask_size = np.size(self.mask_image)
        if (raw_size == mask_size) :
            self.image_width, self.image_height = raw_size
        else:
            print("ERROR: raw and maks image sizes differ")

    def get_raw(self):
        return self.raw_image

    def get_mask(self):
        return self.mask_image
