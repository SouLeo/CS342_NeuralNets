import torch.nn as nn

class MainLinear(nn.Module):
    def __init__(self, input_dim):
    
        super(MainLinear, self).__init__()
       
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            linear_transform = self.linear(x)
            return linear_transform 

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
