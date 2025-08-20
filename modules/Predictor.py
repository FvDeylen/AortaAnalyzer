import numpy as np 

import torch 
import torch.nn.functional as F
import skimage.measure as measure
from skimage import morphology
from PyQt6.QtCore import pyqtSignal
#from torch.utils.tensorboard import SummaryWriter
# internal imports 
from modules.Runet import RUNet
from defaults import *

# TODO: change interpolation/resampling method 
class SegmentationPredictor():
    """
    Wrapper object to call segmentation predictions based on trained RUNet.
    """
    # Training and inferrence with modified version of: https://github.com/MWod/SEGA_MW_2023/tree/main
    progress = pyqtSignal(int,str)
    result = pyqtSignal(object)
    def __init__(self):
        # set model 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("CNN with",self.device)
        self.model = RUNet(**self.__network_config()).to(self.device)
        #input_tensor = torch.randn(16, 1, 400, 400, 400)
        #writer = SummaryWriter(log_dir="C:/Users/abeef/Desktop")
        #writer.add_graph(self.model,input_tensor)
        #writer.close()
        self.trained = torch.load("best_model497", map_location=torch.device(self.device), weights_only=True)   # NG: best_model478_NG
        self.model.load_state_dict(self.trained['model_state_dict'])
        self.model = self.model.eval().to(self.device)
        
        # set additional parameters for inference 
        self.postprocess = False
        self.output_size = (400,400,400)     
        self.windowing = False
        
    def __network_config(self):
       # define network parameters
        input_channels = [1, 6, 16, 64, 128, 256]
        output_channels = [6, 16, 64, 128, 256, 512]
        blocks_per_encoder_channel = [1, 1, 1, 2, 2, 2]
        blocks_per_decoder_channel = [1, 1, 1, 2, 2, 2]
        use_sigmoid = True
        
        # parse parameters
        config = {}
        config['input_channels'] = input_channels
        config['output_channels'] = output_channels
        config['blocks_per_encoder_channel'] = blocks_per_encoder_channel
        config['blocks_per_decoder_channel'] = blocks_per_decoder_channel
        config['img_size'] = None
        config['use_sigmoid'] = use_sigmoid
        return config 
    
    def resample(self, tensor, new_size, device, mode='bilinear'):
        # from preprocessing volumetric
        identity_transform = torch.eye(len(new_size)-1, device=device)[:-1, :].unsqueeze(0)
        identity_transform = torch.repeat_interleave(identity_transform, new_size[0], dim=0)
        sampling_grid = F.affine_grid(identity_transform, new_size, align_corners=False)
        resampled_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)
        return resampled_tensor
        
    
    def run_inferrence(self, volume): 
        # input: path to volume 
        # output: prediction in form of numpy array 
        # format input
        volume = volume.swapaxes(0, 1) 
        lw = -700
        uw = 2300
        
        # inference 
        with torch.set_grad_enabled(False):
            self.progress.emit(0,"Loading Volume ...")
            volume_tc = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            if self.windowing: 
                volume_tc[volume_tc > uw] = uw
                volume_tc[volume_tc < lw] = lw
                volume_tc = volume_tc * (uw-lw)+lw
            volume_tc = (volume_tc - torch.min(volume_tc)) / (torch.max(volume_tc) - torch.min(volume_tc))  # normalize 
            self.progress.emit(1,"Resampling volume to input shape...")
            original_shape = volume_tc.shape
            volume_tc = self.resample(volume_tc, (1, 1, *self.output_size),self.device)  # resample to input size of model 
            self.progress.emit(2,"Generating prediction ...")
            output_tc = self.model(volume_tc)
            self.progress.emit(3,"Resampling prediction to original shape ...")
            output_tc = self.resample(output_tc, original_shape,self.device)  
            self.progress.emit(4,"Converting prediction to numpy array ...")
            prediction = (output_tc[0, 0, :, :, :] > 0.5).detach().cpu().numpy()
            prediction = np.transpose(prediction,(1,0,2))  
            
        # postprocessing
        if self.postprocess:
            self.progress.emit(5, "Postprocessing ...")
            prediction = prediction.astype(np.uint8)
            prediction = morphology.closing(prediction)  # close small gaps
         # remove small clusters 
            label_img = morphology.label(prediction,connectivity=2)
            label_hist , _ = np.histogram(label_img, bins = np.max(label_img)+1)
            for i in range (1, len(label_hist)):
                cluster_size = label_hist[i]
                if 0 < cluster_size < MIN_CLUSTER_SIZE:  
                    prediction[label_img==i] = 0
            prediction = morphology.opening(prediction)  # remove spikes 

        self.result.emit(prediction)
        


    