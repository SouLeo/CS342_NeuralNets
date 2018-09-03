import torch

class Main(torch.nn.Module):
    def forward(self, x):
        # The input x is a series of random numbers of size k x 2
        # You should use these random numbers to compute and return pi using pytorch

#        print('original vector:')
#        print(x)
        x = x[:,0]**2 + x[:,1]**2
#        print('new vector:')
#        print(x)
        x = torch.trunc(x) 
        #x_thresholded = torch.where(x < 1.0, torch.Tensor([1.0]), torch.Tensor([0.0]))
#        print('threshold vector')
#        print(x)
        x = (1 - torch.mean(x)) * 4
#        print('estimate of pi')
#        print(x)
        return x
