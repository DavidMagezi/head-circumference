# %%[markdown]
# External packages 
# %%
import matplotlib.pyplot as plt
from pathlib import Path
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import sys
import torch
from torch.utils.data import DataLoader #, Dataset 

# %%
bjarten_dir = Path('early_stopping')
if bjarten_dir.is_dir():
    sys.path.insert(0,bjarten_dir)

# %%
from early_stopping.pytorchtools import * 
# %%


# %%
class UNet():
    def __init__(self,train_data):
        self.model = smp.Unet(encoder_name="resnet34",
                              encoder_weights="imagenet",
                              in_channels=1,
                              classes=1,
                              activation = "sigmoid")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 0.001
        self.epochs = 1 #50
        self.metrics = [smp.utils.metrics.IoU()]
        self.loss_function = smp.utils.losses.DiceLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)
        self.stopper = EarlyStopping(patience=3)
        self.split_rate = 0.8
        self.train_data = train_data
        self.train_ds_len = int(len(self.train_data) * self.split_rate)
        self.valid_ds_len = len(self.train_data) - self.train_ds_len
        self.train_ds, self.valid_ds = torch.utils.data.random_split(self.train_data, (self.train_ds_len, self.valid_ds_len))
        self.train_epoch = smp.utils.train.TrainEpoch(self.model,
                                                  loss=self.loss_function,
                                                  optimizer=self.optimizer,
                                                  metrics=self.metrics,
                                                  device=self.device,
                                                  verbose=True)
        self.val_epoch = smp.utils.train.ValidEpoch(self.model,
                                                  loss=self.loss_function,
                                                  metrics=self.metrics,
                                                  device=self.device,
                                                  verbose=True)

        self.batch_size = 50 # 
        self.shuffle_training = True #
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle_training)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=self.shuffle_training)
        self.set_figure_dir()

    def set_figure_dir(self,figure_dir='figures'):
        self.figure_dir=Path(figure_dir)
    

    def confirm_loading(self):
        print("This is a test")
        print(f'Train dataset length: {len(self.train_ds)}\n')
        print(f'Validation dataset length: {len(self.valid_ds)}\n')
        print(f'All data length: {len(self.train_data)}\n')

    def train(self): 
        self.train_loss = []
        self.val_loss = []

        self.train_acc = []
        self.val_acc = []

        for epoch in range(self.epochs):
            # training proccess
            print('\nEpoch: {}'.format(epoch))

            self.train_log = self.train_epoch.run(self.train_dl)
            self.val_log = self.val_epoch.run(self.valid_dl)

            self.scheduler.step()

            self.train_loss.append(self.train_log[self.loss_function.__name__])
            self.val_loss.append(self.val_log[self.loss_function.__name__])

            self.train_acc.append(self.train_log['iou_score']) 
            self.val_acc.append(self.val_log['iou_score'])

            self.stopper(self.val_log[self.loss_function.__name__],self.model)
            if self.stopper.early_stop:
                break

    def save_model_parameters(self,model_filename='./model_parameters.binary'):
        torch.save(self.model.state_dict(),model_filename)

    def load_model_parameters(self,model_filename='./model_parameters.binary'):
        self.model.load_state_dict(torch.load(model_filename))
        self.model.eval()

    #def get_loss_and_accuracy(self):
    #    return(self.train_loss,self.val_loss,self.train_acc,self.val_acc)

    def plot_loss_and_accuracy(self):
        self.losses_fig = plt.figure(figsize=(10, 10))
        plt.plot(range(len(self.train_loss)), self.train_loss, label='tain_loss')
        plt.plot(range(len(self.val_loss)), self.val_loss, label='val_loss')
        plt.legend()
        plt.title('Train and validation losses for each epoch', fontdict={'fontsize': 30,}, pad=20)
        self.losses_fig.savefig(self.figure_dir.joinpath('losses_for_each_epoch.png'))

        # %%
        self.accuracy_fig = plt.figure(figsize=(10, 10))
        plt.plot(range(len(self.train_acc)), self.train_acc, label='train_acc')
        plt.plot(range(len(self.val_acc)), self.val_acc, label='val_acc')
        plt.legend()
        plt.title('Train and validation accuracy for each epoch', fontdict={'fontsize': 30,}, pad=20)
        self.accuracy_fig.savefig(self.figure_dir.joinpath('accuracy_for_each_epoch.png'))

    def get_validation_data_item(self,valid_item=0,image_size=(128,128)):
        if 0 <= valid_item < len(self.valid_ds):
            v_image,v_mask = self.valid_ds.__getitem__(valid_item)
            valid_idx = self.valid_ds.indices[valid_item]
            pred_v_t,pred_v_rounded = self.prepare_image_for_plots(v_image,image_size)
        else:
            print("ERROR: index out of bounds")

        return pred_v_t,pred_v_rounded,valid_idx

    def prepare_image_for_plots(self,image,image_size):
        if torch.cuda.is_available():
            image = image.cuda()

        image_4D = torch.reshape(image,(1,1,)+image_size)
        pred = self.model(image_4D)
        pred_numpy = pred.cpu().detach().numpy()
        pred_t = np.transpose(pred_numpy[0], (1, 2, 0))
        pred_rounded = pred[0].squeeze().cpu().detach().numpy().round().astype(np.uint8)
        return pred_t,pred_rounded

    def set_test_data(self,test_data):
        self.test_data = test_data

    def get_test_data_item(self,test_item=0,image_size=(128,128)):
        if 0 <= test_item < len(self.test_data):
            test_image = self.test_data.__getitem__(test_item)
            test_t,test_rounded = self.prepare_image_for_plots(test_image,image_size)
        else:
            print("ERROR: index out of bounds")

        return test_t,test_rounded

    def get_valid_len(self):
        return self.valid_ds_len

    def get_test_len(self):
        return len(self.test_data)


