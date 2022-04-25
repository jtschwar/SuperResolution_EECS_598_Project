from torch import nn
import math

class SRCNN(nn.Module):
    def __init__(self, num_channels=1, filters=[64,32], kernels=[9,5]):
        super(SRCNN, self).__init__()

        # patch extraction
        self.features = nn.Sequential(
                         nn.Conv2d(num_channels,filters[0],kernel_size=kernels[0],padding=kernels[0]//2),
                         nn.ReLU() )
        
        # non-linear mapping
        self.map = nn.Sequential(nn.Conv2d(filters[0],filters[1],kernel_size=kernels[1],padding=kernels[1]//2),
                                 nn.ReLU() )
        
        # reconstruction
        self.recon = nn.Conv2d(filters[1], num_channels,kernel_size=kernels[1],padding=kernels[1]//2) 

        self._initialize_weights()

    def forward(self,x):
        out = self.features(x)
        out = self.map(out)
        out = self.recon(out)
        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, mean=0.0, std=math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels, d=56,s=12,m=4):
        super(FSRCNN, self).__init__()
        
        # feature extraction layer
        self.feature_extraction = nn.Sequential(nn.Conv2d(num_channels, d, kernel_size=5,padding=5//2),
                                                nn.PReLU(d))
    
        # Shrinking Layer
        self.shrink = nn.Sequential(nn.Conv2d(d,s,kernel_size=1), nn.PReLU(s))

        # Mapping Layer
        self.map = []
        for _ in range(m): self.map.extend([nn.Conv2d(s,s,kernel_size=3,padding=3//2), nn.PReLU(s)])
        self.map = nn.Sequential(*self.map)

        # Expanding Layer
        self.expand = nn.Sequential(nn.Conv2d(s,d,kernel_size=1), nn.PReLU(d))

        # Deconvolution Layer
        self.deconv = nn.ConvTranspose2d(d, num_channels, kernel_size=9,stride=scale_factor,
                                         padding=9//2, output_padding=scale_factor-1)

        self._initialize_weights()

    def forward(self,x):
        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)
        return out

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, mean=0.0, std=math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)
    
# Implement EDSR
# class EDSR(nn.Module):
#     def __init__(self,)
