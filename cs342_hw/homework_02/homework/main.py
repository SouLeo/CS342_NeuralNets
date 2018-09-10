import torch.nn as nn

class MainLinear(nn.Module):
    def __init__(self):
    
        super(MainLinear, self).__init__()
      
        self.linear = nn.Linear(2,2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        linear_transform = self.linear(x)
        sigmoid = self.sigmoid(linear_transform)

        return linear_transform, sigmoid
   
class MainDeep(nn.Module):
    def __init__(self):
        super(MainDeep, self).__init__()
        '''
        Your code here
        '''

    def forward(self, x):
        '''
        Your code here
        '''
        return x
