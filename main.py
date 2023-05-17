#!/bin/python

# %%[markdown]
# Copyright (2023) David A. Magezi

# %%[markdown]
# External package for directories
# %%
import numpy as np
from pathlib import Path

# %%[markdown]
# Set directories
# %%
data_dir = Path('..','..','13','medical_data','grand_challenge','us_data_2D') 
figure_dir = Path('figures')
    
# %%[markdown]
# External packages
# %%
from datetime import datetime
import random


# %%[markdown]
# Internal packages
# %%
from ellipse import Ellipse
from fetal_head_dataset import FetalHeadDataset
from mask_to_plot import MaskToPlot
from original_image import OriginalImage
from plot_fetal_head import PlotFetalHead
from print_results import Results
from unet import UNet

# %%
train_dir = data_dir.joinpath('training_set')
test_dir = data_dir.joinpath('test_set')

# %%
if train_dir.is_dir() & test_dir.is_dir():
    train_data = FetalHeadDataset(train_dir)
    test_data = FetalHeadDataset(test_dir,is_test=True)
else:
    print("ERROR: Data directories not found")

show_example_input = True

if show_example_input:
    # %%[markdown]
    # Have a look at randomly selected image
    # %%

    #random_seed = 42
    random_seed = datetime.now().timestamp()
    random.seed(a=random_seed)

    # %%
    #submission_example=413
    example_train_idx = random.randint(0,len(train_data)-1)
    example_train_filename = train_data.get_filename(example_train_idx) 

    # %%
    oi = OriginalImage(train_dir)
    oi.set_image_name(example_train_filename)
    oi.read_images()

    # %%
    pfh = PlotFetalHead('Example Training Image:' + Path(example_train_filename).stem,2)
    original_row = 1

    # %%
    raw_column = 0
    pfh.set_title('raw image',raw_column)
    pfh.set_image_data(train_data.open_as_array(example_train_idx),raw_column)
    pfh.set_title('raw image - original',raw_column,original_row)
    pfh.set_image_data(oi.get_raw(),raw_column,original_row)

    # %%
    mask_column = 1
    pfh.set_title('mask' + ' (' + str(train_data.get_head_circumference(example_train_idx)) + ' mm)',mask_column)
    pfh.set_image_data(train_data.open_mask(example_train_idx),mask_column)
    pfh.set_image_data(oi.get_mask(),mask_column,original_row)

    # %%
    mask_threshold_column = 2
    pfh.set_title('mask threshold',mask_threshold_column)
    mask_threshold_data = MaskToPlot(train_data.open_mask(example_train_idx))
    mask_threshold_data_original = MaskToPlot(oi.get_mask())
    pfh.set_image_data(mask_threshold_data.threshold(),mask_threshold_column)
    pfh.set_image_data(mask_threshold_data_original.threshold(),mask_threshold_column,original_row)

    # %% [markdown]
    # image size for scatter plots ("line plots")
    # %%
    fitted_image_width,fitted_image_height = train_data.get_image_size()
    pfh.set_image_size(fitted_image_width,fitted_image_height)
    fitted_image_width_original = oi.get_image_width()
    fitted_image_height_original = oi.get_image_height()
    pfh.set_image_size(fitted_image_width_original,fitted_image_height_original,original_row)

    # %%
    plot_mask_threshold_column = 3
    pfh.set_title('Plotted mask threshold', plot_mask_threshold_column)
    x_plot_threshold_mask, y_plot_threshold_mask = mask_threshold_data.pixels_to_cartesian()
    x_plot_threshold_mask_original,y_plot_threshold_mask_original = mask_threshold_data_original.pixels_to_cartesian()

    pfh.set_scatter_data(x_plot_threshold_mask,y_plot_threshold_mask,plot_mask_threshold_column)
    pfh.set_scatter_data(x_plot_threshold_mask_original,y_plot_threshold_mask_original,plot_mask_threshold_column,original_row)

    ## %%
    x_example_train,y_example_train = train_data.pixel_to_cart_prediction(train_data.open_mask(example_train_idx))

    # %%
    example_train_head_shape=Ellipse(x_example_train,y_example_train)
    x_fitted_example_train,y_fitted_example_train = example_train_head_shape.fit()
    estimated_head_circumference = example_train_head_shape.estimate_circumference(train_data.get_pixel_size(example_train_idx),train_data.get_original_dimensions(example_train_idx),train_data.get_image_size()) 
    hc = "{:4.1f}".format(estimated_head_circumference)

    # %%
    example_train_head_shape_original=Ellipse(x_plot_threshold_mask_original,y_plot_threshold_mask_original)
    x_fitted_example_train_original,y_fitted_example_train_original = example_train_head_shape_original.fit()
    estimated_head_circumference_original = example_train_head_shape_original.estimate_circumference(train_data.get_pixel_size(example_train_idx),train_data.get_original_dimensions(example_train_idx),train_data.get_original_dimensions(example_train_idx)) 
    hc_original = "{:4.1f}".format(estimated_head_circumference_original)

    # %%
    plot_column = 4
    pfh.set_title('fitted ellipse' + ' (' +  hc + 'mm)',plot_column)
    pfh.set_plot_data(x_fitted_example_train,y_fitted_example_train,plot_column)

    pfh.set_title('fitted ellipse - original' + ' (' +  hc_original + 'mm)',plot_column,original_row)
    pfh.set_plot_data(x_fitted_example_train_original,y_fitted_example_train_original,plot_column,original_row)

    ## %%
    example_train_ellipse_parameters = example_train_head_shape.estimate_parameters(train_data.get_pixel_size(example_train_idx),train_data.get_original_dimensions(example_train_idx),train_data.get_image_size()) 

    #print(example_train_ellipse_parameters)  
    #print(train_data.get_pixel_size(example_train_idx))
    #print(train_data.get_image_size())
    #print(train_data.get_original_dimensions(example_train_idx))

    # %% [markdown]
    # %%
    example_train_ellipse_parameters_original = example_train_head_shape_original.estimate_parameters(train_data.get_pixel_size(example_train_idx),train_data.get_original_dimensions(example_train_idx),train_data.get_original_dimensions(example_train_idx)) 

    # %%
    example_train_results = Results(False)
    example_train_results.nextFile(example_train_filename, example_train_ellipse_parameters)
    example_train_results.nextFile('original_' + example_train_filename,example_train_ellipse_parameters_original)
    example_train_results.print()

    # %%
    if figure_dir.is_dir():
        pfh.save('example_training')
    # %%
    #input("Press Enter to continue with model ...")


unet = UNet(train_data)
unet.confirm_loading()

#unet.train()
#unet.save_model_parameters()
#unet.plot_loss_and_accuracy()

unet.load_model_parameters()
if show_example_input:
    example_valid_item = random.randint(0,unet.get_valid_len())
    pred_v_t,pred_v_rounded,example_valid_idx = unet.get_validation_data_item(example_valid_item,(fitted_image_width,fitted_image_height))
    example_valid_filename = train_data.get_filename(example_valid_idx) 
    pfh_valid = PlotFetalHead('Example Validation Image:' + Path(example_valid_filename).stem,ncols=6)
    pfh_valid.set_image_size(fitted_image_width,fitted_image_height)

    # %%
    raw_column = 0
    pfh_valid.set_title('raw image',raw_column)
    pfh_valid.set_image_data(train_data.open_as_array(example_valid_idx),raw_column)

    # %%
    mask_column = 1
    pfh_valid.set_title('mask' + ' (' + str(train_data.get_head_circumference(example_valid_idx)) + ' mm)',mask_column)
    pfh_valid.set_image_data(train_data.open_mask(example_valid_idx),mask_column)

    # %%
    predicted_mask_column = 2
    pfh_valid.set_title('predicted mask',predicted_mask_column)
    pfh_valid.set_image_data(pred_v_t,predicted_mask_column)

    # %%
    predicted_mask_hist_column = 3
    pfh_valid.set_title('predicted mask histogram',predicted_mask_hist_column)
    pfh_valid.set_hist_data(pred_v_rounded,predicted_mask_hist_column)

    # %%
    mask_threshold_column = 4
    pfh_valid.set_title('predicted mask thresholded',mask_threshold_column)
    pred_mask_threshold_data = MaskToPlot(pred_v_rounded)
    pfh_valid.set_image_data(pred_mask_threshold_data.threshold(),mask_threshold_column)

    ## %%
    x_example_valid,y_example_valid = pred_mask_threshold_data.pixels_to_cartesian()

    # %%
    example_valid_head_shape=Ellipse(x_example_valid,y_example_valid)
    x_fitted_example_valid,y_fitted_example_valid = example_valid_head_shape.fit()
    estimated_head_circumference_valid = example_valid_head_shape.estimate_circumference(train_data.get_pixel_size(example_valid_idx),train_data.get_original_dimensions(example_valid_idx),train_data.get_image_size()) 
    hc_valid = "{:4.1f}".format(estimated_head_circumference_valid)
    
    # %%
    plot_column = 5
    pfh_valid.set_title('fitted ellipse' + ' (' +  hc_valid + 'mm)',plot_column)
    pfh_valid.set_plot_data(x_fitted_example_valid,y_fitted_example_valid,plot_column)

    ## %%
    example_valid_ellipse_parameters = example_valid_head_shape.estimate_parameters(train_data.get_pixel_size(example_valid_idx),train_data.get_original_dimensions(example_valid_idx),train_data.get_image_size()) 

    #print(example_valid_ellipse_parameters)  

    # %%
    example_valid_results = Results(False)
    example_valid_results.nextFile(example_valid_filename, example_valid_ellipse_parameters)
    example_valid_results.print()

    # %%
    if figure_dir.is_dir():
        pfh_valid.save('example_validation')
    # %%

if show_example_input:
    unet.set_test_data(test_data)
    example_test_idx = random.randint(0,unet.get_test_len())
    test_t,test_rounded = unet.get_test_data_item(example_test_idx,(fitted_image_width,fitted_image_height))

    example_test_filename = test_data.get_filename(example_test_idx) 
    pfh_test = PlotFetalHead('Example Test Image:' + Path(example_test_filename).stem,ncols=3)
    pfh_test.set_image_size(fitted_image_width,fitted_image_height)

    # %%
    raw_column = 0
    pfh_test.set_title('raw image',raw_column)
    pfh_test.set_image_data(test_data.open_as_array(example_test_idx),raw_column)

    # %%
    predicted_mask_column = 1
    pfh_test.set_title('predicted mask',predicted_mask_column)
    pfh_test.set_image_data(test_t,predicted_mask_column)

    # %%
    mask_threshold_column = 2
    pfh_test.set_title('predicted mask thresholded',mask_threshold_column)
    test_mask_threshold_data = MaskToPlot(test_rounded)
    pfh_test.set_image_data(test_mask_threshold_data.threshold(),mask_threshold_column)

    # %%
    if figure_dir.is_dir():
        pfh_test.save('example_test')
    # %%
    input("Press Enter to continue with model ...")

#Test:
#original
#prediction

#Then finally save eliipse parameters

