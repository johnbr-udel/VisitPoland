import numpy as np
import torch

class ForwardImaging:
    def __init__(self, resolution, device = 'cpu', vertical_super_res = False, photons = 20) -> None:
        self.resolution = resolution
        self.photons = photons
        
        self.padding = [0, 4, 5]
        
        self.stride = [2, 3, 6] if vertical_super_res else [1, 3, 6]
            
        self.kernel_size = 11

        self.sigma = 6/2.35

        self.kernel = torch.from_numpy(self.gaussian_kernel(self.kernel_size, self.sigma)).unsqueeze(0).float()
        self.kernel = torch.ones(2 if vertical_super_res else 1, self.kernel_size, self.kernel_size) * self.kernel
        self.kernel = self.kernel.to(device)

    def gaussian_kernel(self, size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1/ (2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
            (size, size)
        )
        return kernel / np.sum(kernel)
    
    def forward_imaging(self, high_res, mask = None):

        high_res = high_res.unsqueeze(1)
        print(high_res.shape)
        tamano = high_res.shape
        high_res = torch.nn.functional.interpolate(high_res, size=[tamano[-3],tamano[-2]*self.resolution,tamano[-1]*self.resolution], mode='nearest')
        
        aggregated = torch.nn.functional.conv3d(high_res, self.kernel.unsqueeze(0).unsqueeze(0), 
                                         stride=self.stride, 
                                         padding=self.padding).squeeze()
        
        aggregated = torch.relu(aggregated)
        aggregated = aggregated / (aggregated.sum(0) + 1e-12)

        if mask is not None:
            distribution = torch.distributions.Poisson(self.photons*aggregated[mask == 1])
        else:
            distribution = torch.distributions.Poisson(self.photons*aggregated)

        return distribution, aggregated
    
    def forward_imaging_multinomial(self, high_res, pattern = None):

        high_res = high_res.unsqueeze(1)
        tamano = high_res.shape
        # print(high_res.shape)
        high_res = torch.nn.functional.interpolate(high_res, size=[tamano[-3],tamano[-2]*self.resolution,tamano[-1]*self.resolution], mode='nearest')
        # print(high_res.shape)
        
        aggregated = torch.nn.functional.conv3d(high_res, self.kernel.unsqueeze(0).unsqueeze(0), 
                                         stride=self.stride, 
                                         padding=self.padding).squeeze()

        if pattern is not None:
            distribution = torch.distributions.Multinomial(self.photons, pattern @ aggregated.reshape(aggregated.shape[0],-1).T + 1e-12)
            return distribution, aggregated
    
        else:
            distribution = torch.distributions.Multinomial(self.photons, aggregated.transpose(0,-1) + 1e-12)
            return distribution, aggregated