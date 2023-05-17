# %%
import matplotlib.pyplot as plt
from pathlib import Path

class PlotFetalHead():
    def __init__(self,suptitle,nrows=1,ncols=5):
        self.suptitle=suptitle
        self.nrows=nrows
        self.ncols=ncols
        self.fig, self.ax = plt.subplots(nrows=self.nrows,ncols=self.ncols, figsize=(20,7))
        self.fig.suptitle(self.suptitle)
        self.image_width=list()
        self.image_height=list()

    def set_image_size(self,image_width,image_height,row_index=0):
        if (len(self.image_width) == len(self.image_height) == row_index):
            self.image_width.append(image_width)
            self.image_height.append(image_height)
        else:
            print("ERROR: Please set row information in order")

    def get_current_axis(self,col_index,row_index):
        if (self.nrows == 1):
            current_axis = self.ax[col_index]
        elif (self.nrows > 1):
            current_axis = self.ax[row_index][col_index]

        return current_axis
        

    def set_title(self,title,col_index,row_index=0):
        current_axis = self.get_current_axis(col_index,row_index)
        current_axis.set_title(title)

    def set_image_data(self,image_data,col_index,row_index=0):
        current_axis = self.get_current_axis(col_index,row_index)
        current_axis.imshow(image_data)

    def set_hist_data(self,hist_data,col_index,row_index=0):
        current_axis = self.get_current_axis(col_index,row_index)
        current_axis.hist(hist_data)
        current_axis.set_box_aspect(1.0)


    def set_scatter_data(self,x_data,y_data,col_index,row_index=0):
        current_axis = self.get_current_axis(col_index,row_index)
        if ((len(self.image_width) == len(self.image_height)) & (len(self.image_width)> row_index)):
            aspect_ratio = self.image_height[row_index]/self.image_width[row_index]
            current_axis.scatter(x_data,y_data,marker='.',s=0.5)
            current_axis.set_box_aspect(aspect_ratio)
            if (len(self.image_width) > row_index):
                current_axis.xaxis.set_view_interval(0,self.image_width[row_index])
            if (len(self.image_height) > row_index):
                current_axis.yaxis.set_view_interval(0,self.image_height[row_index])
            #current_axis.set_xlim(0,self.image_width[row_index])
            #current_axis.set_ylim(0,self.image_height[row_index])
            current_axis.yaxis.set_inverted(True)
        else:
            print("ERROR: incorrect or non-matching image_width and height")


    def set_plot_data(self,x_data,y_data,col_index,row_index=0):
        current_axis = self.get_current_axis(col_index,row_index)
        if ((len(self.image_width) == len(self.image_width)) & (len(self.image_width)> row_index)):
            aspect_ratio = self.image_height[row_index]/self.image_width[row_index]
            current_axis.plot(x_data,y_data)
            current_axis.set_box_aspect(aspect_ratio)
            if (len(self.image_width) > row_index):
                current_axis.xaxis.set_view_interval(0,self.image_width[row_index])
            if (len(self.image_height) > row_index):
                current_axis.yaxis.set_view_interval(0,self.image_height[row_index])
            current_axis.set_xlim(0,self.image_width[row_index])
            current_axis.set_ylim(0,self.image_height[row_index])
            current_axis.yaxis.set_inverted(True)

    def save(self,file_stem,figure_dir=Path('figures'),file_extension='png'):
        self.fig.show()
        self.fig.savefig(figure_dir.joinpath(file_stem + file_extension))
