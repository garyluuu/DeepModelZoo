import torch
import torch.nn as nn

class ResMLP(nn.Module):
    def __init__(self, 
                 infeatures: int,
                 hiddenDim: int = None
                ) -> None:
        super().__init__()
        if hiddenDim is None:
            hiddenDim = infeatures
        self.mlp = nn.Sequential(
            nn.LayerNorm(infeatures),
            nn.Linear(infeatures, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, infeatures),
        )
    def forward(self, x):
        return x + self.mlp(x)


class mlp(nn.Module):
    def __init__(self, 
                 inFeatures:list, 
                 hasParams:bool = False
                ):
        super().__init__()
        NumLayers = len(inFeatures)
        self.net = nn.Sequential(
            nn.Linear(inFeatures[0], inFeatures[1]),
            nn.ReLU(),
            *[ResMLP(inFeatures[i],  inFeatures[i+1]) for i in range(1,NumLayers-2)],
            nn.Linear(inFeatures[-2], inFeatures[-1]),
        )

        # self.call = self.__forwardP if hasParams else self.__forward

    def forward(self, x, constant, **kwargs):
        # constant_expanded = constant.repeat(x.shape[0],1,1)
        # x = torch.cat([x, constant_expanded], dim=-1)
        return self.net(x)
    


class mlpford(nn.Module):
    def __init__(self, 
                 inFeatures:list, 
                 hasParams:bool = False
                ):
        super().__init__()
        NumLayers = len(inFeatures)
        self.net = nn.Sequential(
            nn.Linear(inFeatures[0], inFeatures[1]),
            nn.ReLU(),
            *[ResMLP(inFeatures[i],  inFeatures[i+1]) for i in range(1,NumLayers-2)],
            nn.Linear(inFeatures[-2], inFeatures[-1]),
        )

        # self.call = self.__forwardP if hasParams else self.__forward

    def forward(self, x, constant, **kwargs):
        constant_expanded = constant.repeat(x.shape[0],1,1)
        x = torch.cat([x, constant_expanded], dim=-1)
        return self.net(x)
    
    # def __forwardP(self, x, params):
    #     return self.net(x)
    
    # def __forward(self, x, *args, **kwargs):
    #     return self.net(x)


# class LSTM(nn.Module):
#     def __init__(self,
#                  lstmFeatures: list,
#                  mlpFeatures: list,
#                  mlpTemFeatures: list,
#                  hasParams:bool = False
#                 ):
#         super().__init__()
#         self.mlpFeatures = mlpFeatures
#         self.hiddenDim = lstmFeatures[1]
#         self.lstmlayers = lstmFeatures[2]
#         self.lstm = nn.LSTM(lstmFeatures[0], lstmFeatures[1],lstmFeatures[2])
#         # self.lstm = nn.LSTM(lstmFeatures[0],lstmFeatures[-1])
#         self.mlp = mlp(mlpFeatures, hasParams=hasParams)
#         self.mlpTem = mlp(mlpTemFeatures, hasParams=hasParams)

#     # def forward(self, inp, constant, h, c, **kwargs):
#     def forward(self, inp, constant, **kwargs):
#         constant_expanded = constant.repeat(inp.shape[0],1,1)
#         inp = torch.cat([inp, constant_expanded], dim=-1)
#         # inp = inp.reshape(inp.shape[0], -1)
#         # h = torch.zeros(inp.shape[1], self.hiddenDim).to(inp.device)
#         # c = torch.zeros(inp.shape[1], self.hiddenDim).to(inp.device)

#         h = torch.zeros(self.lstmlayers, inp.shape[1], self.hiddenDim).to(inp.device)
#         c = torch.zeros(self.lstmlayers, inp.shape[1], self.hiddenDim).to(inp.device)
#         # h = torch.zeros(1, inp.shape[1], self.hiddenDim).to(inp.device)
#         # c = torch.zeros(1, inp.shape[1], self.hiddenDim).to(inp.device)

#         # hidden_layers = []
#         # for i in range(inp.size()[0]):
#         #     # h, c = self.lstm(inp[i], (h, c))
#         #     # hidden_layers.append(h)
#         #     output, (h, c) = self.lstm(inp[i].unsqueeze(0), (h, c))
#         #     hidden_layers.append(output.squeeze(0))
#         hidden_layers, (h, c) = self.lstm(inp, (h, c))
#         # hidden_layers = torch.stack(hidden_layers, dim=0)

#         # constant_expanded_mlp = constant.repeat(hidden_layers.shape[0],1,1)
#         # u = self.mlp(hidden_layers[-1:,:,:], constant, **kwargs)

#         u = self.mlp(hidden_layers, constant, **kwargs)
#         # u = self.mlp(h, constant, **kwargs) #try to pass h
#         u = u.transpose(0,-1)
#         u = self.mlpTem(u, constant, **kwargs)
#         u = u.transpose(0,-1)

#         # return u, (h, c)
#         return u

    

class LSTM(nn.Module):
    def __init__(self,
                 lstmFeatures: list,
                 mlpFeatures: list,
                 mlpTemFeatures: list,
                 mlpfordFeatures: list,
                 hasParams:bool = False
                ):
        super().__init__()
        self.mlpFeatures = mlpFeatures
        self.hiddenDim = lstmFeatures[1]
        self.lstmlayers = lstmFeatures[2]
        self.lstm = nn.LSTM(lstmFeatures[0], lstmFeatures[1],lstmFeatures[2])
        # self.lstm = nn.LSTM(lstmFeatures[0],lstmFeatures[-1])
        self.mlp = mlp(mlpFeatures, hasParams=hasParams)
        self.mlpTem = mlp(mlpTemFeatures, hasParams=hasParams)
        self.mlpford = mlpford(mlpfordFeatures, hasParams=hasParams)

    # def forward(self, inp, constant, h, c, **kwargs):
    def forward(self, inp, constant, **kwargs):
        # constant_expanded = constant.repeat(inp.shape[0],1,1)
        # inp = torch.cat([inp, constant_expanded], dim=-1)
        # inp = inp.reshape(inp.shape[0], -1)
        # h = torch.zeros(inp.shape[1], self.hiddenDim).to(inp.device)
        # c = torch.zeros(inp.shape[1], self.hiddenDim).to(inp.device)

        hist = inp[:-1,:,:] #history
        init = inp[-1:,:,:] #initial condition


        h = torch.zeros(self.lstmlayers, inp.shape[1], self.hiddenDim).to(inp.device)
        c = torch.zeros(self.lstmlayers, inp.shape[1], self.hiddenDim).to(inp.device)
        # h = torch.zeros(1, inp.shape[1], self.hiddenDim).to(inp.device)
        # c = torch.zeros(1, inp.shape[1], self.hiddenDim).to(inp.device)

        # hidden_layers = []
        # for i in range(inp.size()[0]):
        #     # h, c = self.lstm(inp[i], (h, c))
        #     # hidden_layers.append(h)
        #     output, (h, c) = self.lstm(inp[i].unsqueeze(0), (h, c))
        #     hidden_layers.append(output.squeeze(0))
        hidden_layers, (h, c) = self.lstm(hist, (h, c))
        # hidden_layers = torch.stack(hidden_layers, dim=0)

        # constant_expanded_mlp = constant.repeat(hidden_layers.shape[0],1,1)
        # u = self.mlp(hidden_layers[-1:,:,:], constant, **kwargs)

        h_hist = self.mlp(hidden_layers, constant, **kwargs)
        # u = self.mlp(h, constant, **kwargs) #try to pass h
        h_hist = h_hist.transpose(0,-1)
        h_hist = self.mlpTem(h_hist, constant, **kwargs)
        h_hist = h_hist.transpose(0,-1)

        input = torch.cat([h_hist, init], dim=-1)
        u = self.mlpford(input, constant, **kwargs)


        # return u, (h, c)
        return u
    



class CNN(nn.Module):
    def __init__(self, 
                 inFeatures:list, 
                 hasParams:bool = False
                 ):
        super(CNN, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=inFeatures[0], out_channels=inFeatures[1], kernel_size=inFeatures[2], stride=inFeatures[3], padding=inFeatures[4]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(16 * (X // 2) * (Y // 2), num_classes)  
        )
    
    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(out.size(0), -1)  
        out = self.fc_layer(out)
        return out