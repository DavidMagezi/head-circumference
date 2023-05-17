import numpy as np
class MaskToPlot():
    def __init__(self,pixel_data):
        self.pixel_data=pixel_data
        self.maximum_value=1
        self.minimum_value=0
        self.threshold_value=0.5
        self.initial_pixel_max=np.max(self.pixel_data)

    def threshold(self):
        pixel_data = np.array(self.pixel_data)
        pixel_data = np.where(pixel_data==self.initial_pixel_max, self.maximum_value, self.minimum_value)
        self.thresholded_pixel_data = pixel_data > self.threshold_value
        return self.thresholded_pixel_data

    def pixels_to_cartesian(self):
        self.image_width,self.image_height = np.shape(self.thresholded_pixel_data)
        image_size = (self.image_width,self.image_height)
        number_pixels = np.prod(image_size)
        data_bool = np.reshape(self.thresholded_pixel_data,number_pixels)
        data_ints = np.ones(number_pixels)
        data_ints[data_bool == False] = 0
        #plt.hist(data_ints)
        data_pixels = np.reshape(data_ints,image_size)
        rows_image = np.array([])
        columns_image = np.array([])
        for row_ind in range(self.image_height): 
            for col_ind in range(self.image_width): 
                if data_pixels[col_ind,row_ind] == self.maximum_value:
                    rows_image = np.append(rows_image,row_ind)
                    columns_image = np.append(columns_image,col_ind)
        return (rows_image,columns_image)
