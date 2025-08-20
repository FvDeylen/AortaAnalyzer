import math 
import torch 
import torch.nn.functional as F


# model from https://github.com/MWod/SEGA_MW_2023/tree/main -> assembled methods from runet, and building blocks 

class RUNet(torch.nn.Module):
    def __init__(self, input_channels, output_channels, blocks_per_encoder_channel, blocks_per_decoder_channel, img_size=None, number_of_output_channels=1, use_sigmoid=True):
        super(RUNet, self).__init__()    
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.blocks_per_encoder_channel = blocks_per_encoder_channel
        self.blocks_per_decoder_channel = blocks_per_decoder_channel
        self.number_of_output_channels = number_of_output_channels
        self.image_size = img_size
        self.use_sigmoid = use_sigmoid
        
        if len(self.input_channels) != len(self.output_channels):
            raise ValueError("Number of input channels must be equal to the number of output channels.")
        
        self.encoder = RUNetEncoder(self.input_channels, self.output_channels, self.blocks_per_encoder_channel)
        self.decoder = RUNetDecoder(self.input_channels, self.output_channels, self.blocks_per_decoder_channel)
        if self.use_sigmoid:
            self.last_layer = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=self.output_channels[0], out_channels=self.number_of_output_channels, kernel_size=1),
                torch.nn.Sigmoid()
            )
        else:
            self.last_layer = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=self.output_channels[0], out_channels=self.number_of_output_channels, kernel_size=1),
            )            
        
    def forward(self, x):
        _, _, d, h, w = x.shape
        if self.image_size is not None and (d, h, w) != (self.image_size[0], self.image_size[1], self.image_size[2]):
            x = F.interpolate(x, self.image_size, mode='trilinear')
            
        embeddings = self.encoder(x)
        decoded = self.decoder(embeddings)
        if decoded.shape != x.shape:
            decoded = self.decoder.pad(decoded, x)
        result = self.last_layer(decoded)
        
        if self.image_size is not None and (d, h, w) != (self.image_size[0], self.image_size[1], self.image_size[2]):
            result = F.interpolate(result, (d, h, w), mode='trilinear')
        return result
    
    


class RUNetEncoder(torch.nn.Module):
    def __init__(self, input_channels, output_channels, blocks_per_channel):
        super(RUNetEncoder, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.blocks_per_channel = blocks_per_channel
        self.num_channels = len(input_channels)
        
        if len(self.input_channels) != len(self.output_channels):
            raise ValueError("Number of input channels must be equal to the number of output channels.")
        
        for i, (ic, oc, bpc) in enumerate(zip(input_channels, output_channels, blocks_per_channel)):
            module_list = []
            for j in range(bpc):
                if j == 0:
                    module_list.append(ResidualBlock(ic, oc))
                else:
                    module_list.append(ResidualBlock(oc, oc))
            cic = ic if bpc == 0 else oc
            module_list.append(torch.nn.Conv3d(cic, oc, 4, stride=2, padding=1))
            module_list.append(torch.nn.GroupNorm(oc, oc))
            module_list.append(torch.nn.LeakyReLU(0.01, inplace=True))
            layer = torch.nn.Sequential(*module_list)
            setattr(self, f"encoder_{i}", layer)
        
    def forward(self, x):
        embeddings = []
        cx = x
        for i in range(self.num_channels):
            cx = getattr(self, f"encoder_{i}")(cx)
            embeddings.append(cx)
        return embeddings

class RUNetDecoder(torch.nn.Module):
    def __init__(self, input_channels, output_channels, blocks_per_channel):
        super(RUNetDecoder, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.blocks_per_channel = blocks_per_channel
        self.num_channels = len(input_channels)
        
        for i, (ic, oc, bpc) in enumerate(zip(input_channels, output_channels, blocks_per_channel)):
            module_list = []
            coc = oc if i == self.num_channels - 1 else oc + output_channels[i + 1]
            for j in range(bpc):
                if j == 0: 
                    module_list.append(ResidualBlock(coc, oc))
                else:
                    module_list.append(ResidualBlock(oc, oc))
            cic = coc if bpc == 0 else oc
            module_list.append(torch.nn.ConvTranspose3d(cic, oc, 4, stride=2, padding=1))
            module_list.append(torch.nn.GroupNorm(oc, oc))
            module_list.append(torch.nn.LeakyReLU(0.01, inplace=True))
            layer = torch.nn.Sequential(*module_list)
            setattr(self, f"decoder_{i}", layer)       

    def pad(self,image, template):
        pad_x = math.fabs(image.size(3) - template.size(3))
        pad_y = math.fabs(image.size(2) - template.size(2))
        pad_z = math.fabs(image.size(4) - template.size(4))
        b_x, e_x = math.floor(pad_x / 2), math.ceil(pad_x / 2)
        b_y, e_y = math.floor(pad_y / 2), math.ceil(pad_y / 2)
        b_z, e_z = math.floor(pad_z / 2), math.ceil(pad_z / 2)
        image = F.pad(image, (b_z, e_z, b_x, e_x, b_y, e_y))
        return image

    def forward(self, embeddings):
        for i in range(self.num_channels - 1, -1, -1):
            if i == self.num_channels - 1:
                cx = getattr(self, f"decoder_{i}")(embeddings[i])         
            else:
                cx = getattr(self, f"decoder_{i}")(torch.cat((self.pad(cx, embeddings[i]), embeddings[i]), dim=1))       
        return cx
    
    
        
class ResidualBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, leaky_alpha=0.01):
        super(ResidualBlock, self).__init__()

        self.module = torch.nn.Sequential(
            torch.nn.Conv3d(input_size, output_size, 3, stride=1, padding=1),
            torch.nn.GroupNorm(output_size, output_size),
            torch.nn.LeakyReLU(leaky_alpha, inplace=True),
            torch.nn.Conv3d(output_size, output_size, 3, stride=1, padding=1),
            torch.nn.GroupNorm(output_size, output_size),
            torch.nn.LeakyReLU(leaky_alpha, inplace=True),        
        )

        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(input_size, output_size, 1)
        )

    def forward(self, x : torch.Tensor):
        return self.module(x) + self.conv(x)