# %%[markdown]
# Copyright (2023) David A. Magezi
# %%
data_dir = Path('..','13','medical_data','grand_challenge','us_data_2D') 

# %%[markdown]
# Bibliotheken
import cv2
from ellipse import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import random
import re
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import sys
import torch
from torch.utils.data import Dataset, DataLoader 
import torchvision

# %%
bjarten_dir = Path('early_stopping')
if bjarten_dir.is_dir():
    sys.path.insert(0,bjarten_dir)

# %%
from early_stopping.pytorchtools import * 
# %%

# %%
figure_dir = Path('figures')


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
                
        print(np.shape(self.image_files))
        print(np.shape(self.pixel_sizes))
        if self.is_test:
            original_df = pd.DataFrame([self.image_files,self.pixel_sizes])
            original_df = original_df.transpose()
            original_columns = ['filename',*structured_info['structured_columns']]
            original_df.columns = original_columns
            print(original_df.head())
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
        number_pixels = np.prod(data.get_image_size())
        example_bool = np.reshape(example_bool,number_pixels)
        example_ints = np.ones(number_pixels)
        example_ints[example_bool == False] = 0
        #plt.hist(example_ints)
        example_pixels = np.reshape(example_ints,data.get_image_size())
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
        number_pixels = np.prod(data.get_image_size())
        example_bool = np.reshape(example_bool,number_pixels)
        example_ints = np.ones(number_pixels)
        example_ints[example_bool == False] = 0
        #plt.hist(example_ints)
        example_pixels = np.reshape(example_ints,data.get_image_size())
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


    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())

        return s
    

# %%
train_dir = data_dir.joinpath('training_set')
test_dir = data_dir.joinpath('test_set')

# %%
if train_dir.is_dir() & test_dir.is_dir():
    data = FetalHeadDataset(train_dir)
else:
    print("ERROR: Data directories not found")

# %%

# %%
random_seed = 42
random.seed(a=random_seed)

# %%
example_training_image = random.randint(0,len(data)-1)
fig_train, ax_train = plt.subplots(nrows=1,ncols=3, figsize=(20,7))
fig_train.suptitle('Example Training Image')

ax_train[0].imshow(data.open_as_array(example_training_image))
ax_train[0].set_title('raw image')
ax_train[1].imshow(data.open_mask(example_training_image))
ax_train[1].set_title('mask' + ' (' + str(data.get_head_circumference(example_training_image)) + ' mm)')

x_example_train,y_example_train = data.pixel_to_cart(example_training_image)
head_shape=Ellipse(x_example_train,y_example_train)
x_fitted_example_train,y_fitted_example_train = head_shape.fit()
ax_train[2].plot(x_fitted_example_train,y_fitted_example_train)
ax_train[2].set_box_aspect(1.0)
image_width,image_height = data.get_image_size()
ax_train[2].xaxis.set_view_interval(0,image_width)
ax_train[2].yaxis.set_view_interval(0,image_height)
ax_train[2].yaxis.set_inverted(True)
estimated_head_circumference = head_shape.estimate_circumference(data.get_pixel_size(example_training_image),data.get_original_dimensions(example_training_image),data.get_image_size()) 
hc = "{:4.1f}".format(estimated_head_circumference)
ax_train[2].set_title('fitted ellipse' + ' (' +  hc + 'mm)') 

if figure_dir.is_dir():
    fig_train.savefig(figure_dir.joinpath('example_training_image.png'))


# %% [markdown]
# #Model configuring
# %%
# Model
# Note -> need internet access
unet = model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
    activation = "sigmoid"
)

# %%
# Device type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Params
learning_rate = 0.001
epochs = 50

# %% [markdown]
# Intersection over Union 
metrics = [smp.utils.metrics.IoU()]

# %% [markdown]
# Loss & optimizer
# %%
loss_function = smp.utils.losses.DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %% [markdown]
# Scheduler & stopper
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
stopper = EarlyStopping(patience=3)

# %% [markdown]
# Split into train and validation
split_rate = 0.8
train_ds_len = int(len(data) * split_rate)
valid_ds_len = len(data) - train_ds_len

train_ds, valid_ds = torch.utils.data.random_split(data, (train_ds_len, valid_ds_len))

print(f'Train dataset length: {len(train_ds)}\n')
print(f'Validation dataset length: {len(valid_ds)}\n')
print(f'All data length: {len(data)}\n')

# %%

# %% [markdownn]
# Train & validation functions
# %%
train_epoch = smp.utils.train.TrainEpoch(model,
                                          loss=loss_function,
                                          optimizer=optimizer,
                                          metrics=metrics,
                                          device=device,
                                          verbose=True)
# %%
val_epoch = smp.utils.train.ValidEpoch(model,
                                          loss=loss_function,
                                          metrics=metrics,
                                          device=device,
                                          verbose=True)

# %% [markdown]
# Data loaders
# %%
batch_size = 50 # 
shuffle_training = True #
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_training)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=shuffle_training)

# %% [markdown]
# # Model training
# %%
train_loss = []
val_loss = []

train_acc = []
val_acc = []

# %%
for epoch in range(epochs):
    # training proccess
    print('\nEpoch: {}'.format(epoch))

    train_log = train_epoch.run(train_dl)
    val_log = val_epoch.run(valid_dl)

    scheduler.step()

    train_loss.append(train_log[loss_function.__name__])
    val_loss.append(val_log[loss_function.__name__])

    train_acc.append(train_log['iou_score']) 
    val_acc.append(val_log['iou_score'])

    stopper(val_log[loss_function.__name__], model)
    if stopper.early_stop:
        break

# %%
# Save model paramaters
torch.save(model.state_dict(),'./model_parameters.binary')

# %%
#model.load_state_dict(torch.load('./model_parameters.binary'))
#model.eval()

# %% [markdown]
# Train results
losses_fig = plt.figure(figsize=(10, 10))
plt.plot(range(len(train_loss)), train_loss, label='tain_loss')
plt.plot(range(len(val_loss)), val_loss, label='val_loss')
plt.legend()
plt.title('Train and validation losses for each epoch', fontdict={'fontsize': 30,}, pad=20)
losses_fig.savefig(figure_dir.joinpath('losses_for_each_epoch.png'))

# %%
accuracy_fig = plt.figure(figsize=(10, 10))
plt.plot(range(len(train_acc)), train_acc, label='train_acc')
plt.plot(range(len(val_acc)), val_acc, label='val_acc')
plt.legend()
plt.title('Train and validation accuracy for each epoch', fontdict={'fontsize': 30,}, pad=20)
accuracy_fig.savefig(figure_dir.joinpath('accuracy_for_each_epoch.png'))

# %%
n_plot_columns = np.min([len(valid_ds),5])

fx_valid, ax_valid = plt.subplots(nrows=3, ncols=n_plot_columns, figsize=(10,10))

for i in range(n_plot_columns):
    v_image,v_mask = valid_ds.__getitem__(i)

    np.shape(np.transpose(v_image, (1, 2, 0)))
    valid_idx = valid_ds.indices[i]
    print(valid_idx)
    hc_valid_actual = data.get_head_circumference(valid_idx)
    if n_plot_columns > 1:
        ax_valid[0][i].imshow(np.transpose(v_image, (1, 2, 0)))
        ax_valid[0][i].set_title('original')
        ax_valid[1][i].imshow(np.transpose(v_mask, (1, 2, 0)))
        ax_valid[1][i].set_title('mask' + '(' + str(hc_valid_actual) + ' mm)')
    else:
        ax_valid[0].imshow(np.transpose(v_image, (1, 2, 0)))
        ax_valid[0].set_title('original')
        ax_valid[1].imshow(np.transpose(v_mask, (1, 2, 0)))
        ax_valid[1].set_title('mask' + '(' + str(hc_valid_actual) + ' mm)')

    if torch.cuda.is_available():
        v_image = v_image.cuda()

    # need to create 4D image here
    v_image_4D = torch.reshape(v_image,(1,1,)+data.get_image_size())
    pred_v = unet(v_image_4D)
    pred_v = pred_v.cpu().detach().numpy()

    estimated_head_circumference = head_shape.estimate_circumference(data.get_pixel_size(valid_idx),data.get_original_dimensions(valid_idx),data.get_image_size()) 
    hc_valid_pred = "{:4.1f}".format(estimated_head_circumference)

    if n_plot_columns > 1:
        ax_valid[2][i].imshow(np.transpose(pred_v[0], (1, 2, 0)))
        ax_valid[2][i].set_title('pred' + ' (' +  hc_valid_pred + 'mm)') 
        #ax_valid[2][i].set_box_aspect(1.0)
        #ax_valid[2][i].xaxis.set_view_interval(0,image_width)
        #ax_valid[2][i].yaxis.set_view_interval(0,image_height)
    else:
        ax_valid[2].imshow(np.transpose(pred_v[0], (1, 2, 0)))
        ax_valid[2].set_title('pred' + ' (' +  hc_valid_pred + 'mm)') 
        #ax_valid[2].set_box_aspect(1.0)
        #ax_valid[2].xaxis.set_view_interval(0,image_width)
        #ax_valid[2].yaxis.set_view_interval(0,image_height)


# %%
test_data = FetalHeadDataset(test_dir,is_test=True)
len(test_data)

# %% [markdown]
# Example Image from test
# %%
fig_test,ax_test = plt.subplots(nrows=1,ncols=4,figsize=(20,7))
fig_test.suptitle('Example Test (and Prediction)')

example_test_image = random.randint(0,len(test_data)-1)
ax_test[0].imshow(test_data.open_as_array(example_test_image))
ax_test[0].set_title('test image')

test_image = test_data.__getitem__(example_test_image)
if torch.cuda.is_available():
    test_image = test_image.cuda()
test_image_4D = torch.reshape(test_image,(1,1,)+test_data.get_image_size())

pr_mask = unet(test_image_4D)
pr_mask = pr_mask[0]
pr_mask = pr_mask.squeeze().cpu().detach().numpy().round().astype(np.uint8)

ax_test[1].imshow(pr_mask)
ax_test[1].set_title('predicted mask')
ax_test[2].hist(pr_mask)
ax_test[2].set_title('distribution')
ax_test[2].set_box_aspect(1.0)

x_test,y_test = test_data.pixel_to_cart_prediction(pr_mask)
test_head_shape=Ellipse(x_test,y_test)
x_fitted_test,y_fitted_test = test_head_shape.fit()
ax_test[3].plot(x_fitted_test,y_fitted_test)
ax_test[3].set_box_aspect(1.0)
ax_test[3].xaxis.set_view_interval(0,image_width)
ax_test[3].yaxis.set_view_interval(0,image_height)
ax_test[3].yaxis.set_inverted(True)
estimated_head_circumference_pred = test_head_shape.estimate_circumference(test_data.get_pixel_size(example_test_image),test_data.get_original_dimensions(example_test_image),test_data.get_image_size()) 
hc_pred = "{:4.1f}".format(estimated_head_circumference_pred)
ax_test[3].set_title('fitted ellipse' + ' (' +  hc_pred + 'mm)') 
fig_test.savefig(figure_dir.joinpath('example_test_image.png'))
