# %% [markdown]
# %% Librarires
# %%
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader 

import numpy as np
import pandas as pd
import re
import torch
import torchvision

# %%[markdown]
# Dataset Klasse
# %%
class FetalHeadDataset(Dataset):
    def __init__(self, directory, is_test=False ,pytorch=True,):
        super().__init__()

        self.is_test = is_test
        structured_info = {'structured_filename':'test_set_pixel_size','structured_columns':['pixel size(mm)']} if self.is_test else {'structured_filename':'training_set_pixel_size_and_HC','structured_columns':['pixel size(mm)','head circumference (mm)']}

        structured_extension = '.csv'
        structured_data_file = directory.parent.joinpath(structured_info['structured_filename'] + structured_extension)
        structured_data = pd.read_csv(structured_data_file)

        # Loop through the files in 'directory' folder and combine, into a dictionary, the masks
        self.image_extension='.png'
        self.mask_suffix='_Annotation'
        self.files = []
        self.image_files = []
        self.pixel_sizes = []
        self.head_circumferences = []
        for ind,file_name in enumerate(directory.iterdir()):
            if self.mask_suffix in str(file_name):
                continue
                
            self.files.append(self.combine_files(file_name))
            image_idx = structured_data.loc[:,'filename'] == file_name.name
            curr_structured = structured_data.loc[image_idx,structured_info['structured_columns']].to_numpy()
            curr_pixel_size = curr_structured[0,0]
            self.pixel_sizes.append(curr_pixel_size) 
            self.image_files.append(file_name) 
            if not self.is_test:
                curr_head_circumference = curr_structured[0,1]
                self.head_circumferences.append(curr_head_circumference) 
                
        #print(np.shape(self.image_files))
        #print(np.shape(self.pixel_sizes))
        if self.is_test:
            original_df = pd.DataFrame([self.image_files,self.pixel_sizes])
            original_df = original_df.transpose()
            original_columns = ['filename',*structured_info['structured_columns']]
            original_df.columns = original_columns
            #print(original_df.head())
        else:
            original_df = pd.DataFrame([self.image_files,self.pixel_sizes,self.head_circumferences])
            original_df = original_df.transpose()
            original_columns = ['filename',*structured_info['structured_columns']]
            original_df.columns = original_columns

        self.original_df = original_df

        # Sorting files list
        self.files = sorted(self.files, key=lambda file: int(re.search(r'\d+', str(file['image'])).group(0)))

        self.image_width = pow(2,7) # 2⁷ = 128, 2⁹=512
        self.image_height = self.image_width
        self.resize = torchvision.transforms.Resize((self.image_height,self.image_width),interpolation=Image.NEAREST)
        self.pytorch = pytorch
        
    def combine_files(self, file_name: Path):

        mask_file_name = file_name.parent.joinpath(file_name.stem + self.mask_suffix + self.image_extension)
        
        files = {
            'image': file_name, 
            'mask': mask_file_name
        }

        return files
                                       
    def __len__(self):
        
        return len(self.files)
     
    def open_as_array(self, idx, invert=False):
        unresized_image = Image.open(self.files[idx]['image'])
        raw_image = self.resize(unresized_image)
        raw_image = raw_image = np.stack([ np.array(raw_image) ], axis=2)
    
        if invert:
            raw_image = raw_image.transpose((2,0,1))
    
        # normalize
        return (raw_image / np.iinfo(raw_image.dtype).max)
    

    def open_mask(self, idx, add_dims=False):
        
        raw_mask = self.resize(Image.open(self.files[idx]['mask']))
        raw_mask = np.array(raw_mask)
        raw_mask = np.where(raw_mask==255, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        
        if not self.is_test:    
            y = torch.tensor(self.open_mask(idx, add_dims=True), dtype=torch.torch.float32)
            return x, y
        
        return x
    
    def open_as_pil(self, idx):
        
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'L')
    
    def get_image_size(self):

        return (self.image_width,self.image_height)

    def pixel_to_cart(self,image_number):
        example_data = self.open_mask(image_number)
        example_bool = example_data > 0.5
        number_pixels = np.prod(self.get_image_size())
        example_bool = np.reshape(example_bool,number_pixels)
        example_ints = np.ones(number_pixels)
        example_ints[example_bool == False] = 0
        #plt.hist(example_ints)
        example_pixels = np.reshape(example_ints,self.get_image_size())
        rows_image = np.array([])
        columns_image = np.array([])
        for row_ind in range(self.image_height): 
            for col_ind in range(self.image_width): 
                if example_pixels[col_ind,row_ind] == 1:
                    rows_image = np.append(rows_image,row_ind)
                    columns_image = np.append(columns_image,col_ind)

        return (rows_image,columns_image)

    def pixel_to_cart_prediction(self,pr_mask):
        example_data = pr_mask
        example_bool = example_data > 0.5
        number_pixels = np.prod(self.get_image_size())
        example_bool = np.reshape(example_bool,number_pixels)
        example_ints = np.ones(number_pixels)
        example_ints[example_bool == False] = 0
        #plt.hist(example_ints)
        example_pixels = np.reshape(example_ints,self.get_image_size())
        rows_image = np.array([])
        columns_image = np.array([])
        for row_ind in range(self.image_height): 
            for col_ind in range(self.image_width): 
                if example_pixels[col_ind,row_ind] == 1:
                    rows_image = np.append(rows_image,row_ind)
                    columns_image = np.append(columns_image,col_ind)

        return (rows_image,columns_image)

    def get_files(self):
        return self.files

    def get_pixel_size(self,idx):
        image_idx = self.original_df.loc[:,'filename'] == self.files[idx]['image']
        pixel_size = self.original_df.loc[image_idx,'pixel size(mm)']
        original_dimensions = self.get_original_dimensions(idx)
        return np.float64(pixel_size)

    def get_head_circumference(self,idx):
        if self.is_test:
            return -1
        else:
            image_idx = self.original_df.loc[:,'filename'] == self.files[idx]['image']
            head_circumference = self.original_df.loc[image_idx,'head circumference (mm)']
            return np.float64(head_circumference)
    
    def get_original_dimensions(self,idx):
        unresized_image = Image.open(self.files[idx]['image'])
        return np.shape(unresized_image)

    def get_filename(self,idx):
        return Path(self.files[idx]['image']).name

    def get_idx(self,file_name):
        #image_idx = self.original_df.loc[:,'filename'] == self.files[idx]['image']
        return_idx=-1
        for idx in np.arange(0,len(self.files)-1) :
            curr_name = Path(self.files[idx]['image']).name
            if (curr_name == file_name) :
                return_idx=idx
        return return_idx

    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())

        return s
