import os
import torch
import deepMZ
import numpy as np
import matplotlib.pyplot as plt
import plot_trajectory

from ReadInput import TrainParamReader

from torch.utils.tensorboard import SummaryWriter
logger_path = './logger/LSTM-lorenz'
writer = SummaryWriter(logger_path)


def train(trainloader, config: TrainParamReader, testloader=None):
    os.makedirs(os.path.dirname(config.savePath), exist_ok=True)
    folder_path = os.path.join(config.savePath, f"{config.nn}_{sys.argv[1].split('_')[2].split('.')[0]}_{config.wrapperParams['evolveLen']}evolveLen_{config.wrapperParams['inputFnArgs']['inputLen']}inputLen")
    os.makedirs(folder_path, exist_ok=True)

    net0 = getattr(deepMZ.temporal.nn, config.nn)(**config.nnParams).to(device)
    # evo_net = getattr(deepMZ.temporal.nn, config_mlp.nn)(**config_mlp.nnParams).to(device)
    if config.wrapper: 
        # net = deepMZ.temporal.wrapper(net, evo_net, **config.wrapperParams)#.to(device)
        net = deepMZ.temporal.wrapper(net0, **config.wrapperParams)#.to(device)
    optimizer = getattr(torch.optim, config.optimizer)(net.parameters(), lr=config.lr)
    criteria = torch.nn.MSELoss()
    scheduler = (
        getattr(torch.optim.lr_scheduler, config.scheduler)(
            optimizer, **config.schedulerParams
        )
        if config.scheduler is not None
        else torch.optim.ConstantLR(optimizer, factor=1, total_iters=1)
    )

    loss_list = []  # List to store the losses
    test_loss = []  

    for i in range(config.epochs):
        epochLoss = 0
        net.train()
        for inp, params, label in trainloader:
            inp = inp.transpose(0,1)#.to(device)
            label = label.transpose(0,1)#.to(device) 
            out = net(inp, constant=params[None]) 
            loss = criteria(out, label)
            epochLoss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(epochLoss / len(trainloader))
        scheduler.step(epochLoss)
        if i % config.printInterval == 0:
            print(f"epoch {i}: {epochLoss/len(trainloader)}") 

        # Clear and Save the loss_list to file
        file_name_losslist = f'lstm_loss_list_lr{config.lr}_{config.wrapperParams["evolveLen"]}evolveLen_{config.wrapperParams["inputFnArgs"]["inputLen"]}inputLen_{config.epochs}epoch.txt'
        # os.path.join(folder_path, file_name_losslist)
        if i == 0:
            with open(os.path.join(folder_path, file_name_losslist), 'w') as file:
                file.write('')
        with open(os.path.join(folder_path, file_name_losslist), 'a') as file:
            file.write(f"epoch{i}: {epochLoss/len(trainloader)}\n")

        # test
        if i % config.saveInterval == 0:
            # test_net = getattr(deepMZ.temporal.nn, config.nn)(**config.nnParams).to(device)
            # if config.wrapper:
            #     test_net = deepMZ.temporal.wrapper_test(net, **config.wrapperParams)
            testloss = 0
            testnet = deepMZ.temporal.wrapper(net0, **config.testParams)
            testnet.eval()
            if not testloader is None:
                with torch.no_grad():
                    for inp, params, label in testloader:
                        inp = inp.transpose(0,1)
                        label = label.transpose(0,1)
                        out = testnet(inp, constant=params[None])
                        # out = torch.cat((inp, out), dim=0)
                        # while out.size(0) < label.size(0):
                        #     temp = net(out[-inp.size(0):, :, :], constant=params[None])
                        #     out = torch.cat((out, temp), dim=0)
                        # out = out[:label.size(0),:,:]
                        # print(out)
                        loss = criteria(out, label)
                        testloss += loss.item()
            print(f"test loss: {testloss/len(testloader)}")
            test_loss.append(testloss/len(testloader))
            with open(os.path.join(folder_path, file_name_losslist), 'a') as file:
                file.write(f"test loss: {testloss/len(testloader)}\n")
            torch.save(net.state_dict(), os.path.join(folder_path, f"{config.nn}_model-{config.epochs}epoches@{i}epoch-{config.wrapperParams['evolveLen']}evolveLen_{config.wrapperParams['inputFnArgs']['inputLen']}inputLen)"))

            # Plot the trajectory
            index = 0

            org_data = np.load(config.dataPath)

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(org_data['u'][:,index,0], org_data['u'][:,index,1], org_data['u'][:,index,2], label='original data', marker='.', markersize=2)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.legend()

            org_data_torch={}
            org_data_torch['u'] = torch.from_numpy(org_data['u']).to(device)
            org_data_torch['params'] = torch.from_numpy(org_data['params']).to(device)
            umean = org_data_torch['u'].mean(axis=(0,1),keepdims=True)
            ustd = org_data_torch['u'].std(axis=(0,1),keepdims=True)
            pmean = org_data_torch['params'].mean(axis=0,keepdims=True)
            pstd = org_data_torch['params'].std(axis=0,keepdims=True)
            normed_data={}
            normed_data['u'] = (org_data_torch['u']-umean)/ustd
            normed_data['params'] = (org_data_torch['params']-pmean)/pstd

            # normed_data['u'] = torch.from_numpy(normed_data['u']).to(device)
            # normed_data['params'] = torch.from_numpy(normed_data['params']).to(device)

            predicted_normed_trajectory = plot_trajectory.predict_trajectory(testnet, config, normed_data, index)
            predicted_trajectory = predicted_normed_trajectory*ustd+umean
            predicted_trajectory = predicted_trajectory.cpu().numpy()

            inputLen = config.wrapperParams['inputFnArgs']['inputLen']
            ax.plot(predicted_trajectory[:,0,0], predicted_trajectory[:,0,1], predicted_trajectory[:,0,2], label='predicted data',marker='.' , markersize = 2)
            ax.plot(predicted_trajectory[:inputLen, 0, 0], predicted_trajectory[:inputLen, 0, 1], predicted_trajectory[:inputLen, 0, 2], color='red', marker='*', markersize=5)
            ax.legend()

            writer.add_figure(f'predicted_{sys.argv[1].split("_")[2].split(".")[0]}', fig, global_step=i)
            
    # Plot the loss_list
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(10,config.epochs-1,config.epochs-10),loss_list[10:])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")  # Set y-axis to logarithmic scale
    plt.title("Training Loss")
    plt.savefig(os.path.join(folder_path,f"lstm_loss_lr{config.lr}_{config.wrapperParams['evolveLen']}evolveLen_{config.wrapperParams['inputFnArgs']['inputLen']}inputLen_{config.epochs}epoch.png"))

    # Plot the test_loss
    if config.epochs >= config.saveInterval:
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0,config.epochs-1,config.epochs//config.saveInterval+1),test_loss)
        plt.xlabel("Epoch")
        plt.ylabel("test_loss")
        plt.yscale("log")  # Set y-axis to logarithmic scale
        plt.title("Testing Loss")
        plt.savefig(os.path.join(folder_path,f"lstm_testloss_lr{config.lr}_{config.wrapperParams['evolveLen']}evolveLen_{config.wrapperParams['inputFnArgs']['inputLen']}inputLen_{config.epochs}epoch.png"))

    # torch.save(net.state_dict(), os.path.join(config.savePath, f"model-{config.epochs}-{config.wrapperParams['evolveLen']}"))

# def test(testloader, config: TrainParamReader):
#     # Load the model
#     model_path = os.path.join(config.savePath, f"model-{config.epochs}-{config.wrapperParams['evolveLen']}")
#     net = getattr(deepMZ.temporal.nn, config.nn)(**config.nnParams).to(device)
#     net.load_state_dict(torch.load(model_path))
#     criteria = torch.nn.MSELoss()

#     net.eval()

#     test_loss = 0
#     if not testloader is None:
#         with torch.no_grad():
#             for inp, params, label in testloader:
#                 inp = inp.transpose(0, 1)
#                 label = label.transpose(0, 1)
#                 out = net(inp, constant=params[None])
#                 loss = criteria(out, label)
#                 test_loss += loss.item()
#         test_loss /= len(testloader)
#         print(f"test loss: {test_loss}")
#         with open(f'loss_list_lr{config.lr}_{config.wrapperParams["evolveLen"]}evolveLen_{config.epochs}epoch.txt', 'a') as file:
#             file.write(f"test loss: {test_loss}\n")

class dset(torch.utils.data.Dataset):
    def __init__(self,normed_data):
        
        self.data = torch.from_numpy(normed_data['u']).to(device)
        self.params = torch.from_numpy(normed_data['params']).to(device)

    def __getitem__(self, index):
        evolvelen = config.wrapperParams['evolveLen']
        inputlen = config.wrapperParams['inputFnArgs']['inputLen']
        init_point = index//self.data.shape[1]
        data_num = index % self.data.shape[1]
        # if init_point == 0:
        #     data_num = index
        # else:
        #     data_num = index % init_point
        return self.data[init_point:init_point+inputlen,data_num,:], self.params[data_num], self.data[init_point+inputlen:init_point+inputlen+evolvelen,data_num,:]

    def __len__(self):
        evolvelen = config.wrapperParams['evolveLen']
        inputlen = config.wrapperParams['inputFnArgs']['inputLen']
        # return self.data.shape[1]*(self.data.shape[0]-inputlen-evolvelen*inputlen)
        return self.data.shape[1]*(self.data.shape[0]-inputlen-evolvelen)
    
class test_dset(torch.utils.data.Dataset):
    def __init__(self,normed_data):
        
        self.data = torch.from_numpy(normed_data['u']).to(device)
        self.params = torch.from_numpy(normed_data['params']).to(device)

    def __getitem__(self, index):
        # testLen = config.testLen
        inputlen = config.wrapperParams['inputFnArgs']['inputLen']
        # init_point = index//self.data.shape[1]
        # data_num = index % self.data.shape[1]
        # return self.data[init_point:init_point+1,data_num,:], self.params[data_num], self.data[init_point+1:init_point+1+testLen*inputlen,data_num,:]
        return self.data[:inputlen,index,:], self.params[index], self.data[inputlen:,index,:]

    def __len__(self):
        # testLen = config.testLen
        # inputlen = config.wrapperParams['inputFnArgs']['inputLen']
        # return self.data.shape[1]*(self.data.shape[0]-testLen*inputlen)
        return self.data.shape[1]


if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # config = TrainParamReader(sys.argv[1])
    # org_data = np.load(config.dataPath)
    # umean = org_data['u'].mean(axis=-1,keepdims=True)
    # ustd = org_data['u'].std(axis=-1,keepdims=True)
    # pmean = org_data['params'].mean(axis=-1,keepdims=True)
    # pstd = org_data['params'].std(axis=-1,keepdims=True)
    # normed_data={}
    # normed_data['u'] = (org_data['u']-umean)/ustd
    # normed_data['params'] = (org_data['params']-pmean)/pstd
    # dataloader = DataLoader(dset(normed_data), batch_size=config.batchSize, shuffle=True)

    config = TrainParamReader(sys.argv[1])
    org_data = np.load(config.dataPath)
    umean = org_data['u'].mean(axis=(0,1),keepdims=True)
    ustd = org_data['u'].std(axis=(0,1),keepdims=True)
    pmean = org_data['params'].mean(axis=0,keepdims=True)
    pstd = org_data['params'].std(axis=0,keepdims=True)
    normed_data={}
    normed_data['u'] = (org_data['u']-umean)/ustd
    normed_data['params'] = (org_data['params']-pmean)/pstd

    # Split the dataset into training and testing
    train_ratio = 0.9
    train_size = int(train_ratio * len(normed_data['params'])) 
    test_size = len(normed_data['params']) - train_size
    train_indices = np.random.choice(len(normed_data['params']), train_size, replace=False)
    test_indices = np.setdiff1d(np.arange(len(normed_data['params'])), train_indices)

    train_data = {}
    train_data['u'] = normed_data['u'][:, train_indices, :]
    train_data['params'] = normed_data['params'][train_indices]

    test_data = {}
    test_data['u'] = normed_data['u'][:, test_indices, :]
    test_data['params'] = normed_data['params'][test_indices]

    train_dataset = dset(train_data)
    test_dataset = test_dset(test_data)

    # Set up dataloader
    trainloader = DataLoader(train_dataset, batch_size=config.batchSize, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=config.batchSize, shuffle=False)

    train(trainloader, config, testloader=testloader)

    plt.figure(figsize=(10, 6))

    plt.scatter(train_indices, np.zeros_like(train_indices), color='blue', label='Train Indices', marker='o')
    plt.scatter(test_indices, np.zeros_like(test_indices), color='red', label='Test Indices', marker='x')
    plt.xlim(0,len(normed_data['params']))
    plt.legend()

    writer.add_figure(f'index_distribution_{sys.argv[1].split("_")[2].split(".")[0]}', plt.gcf())
    train_indices_text = 'Train Indices: ' + ', '.join(map(str, train_indices))
    test_indices_text = 'Test Indices: ' + ', '.join(map(str, test_indices))
    writer.add_text('Indices Text', train_indices_text + '\n' + test_indices_text)

    writer.close()
    # # test(testloader, config)


    # # Set up the whole dataset
    # full_dataset = dset(normed_data)

    # # Split the dataset into training and testing
    # train_ratio = 0.9
    # train_size = int(train_ratio * len(full_dataset))  # Use {train_ratio} of data as training data
    # test_size = len(full_dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # # Set up dataloader
    # trainloader = DataLoader(train_dataset, batch_size=config.batchSize, shuffle=True)
    # testloader = DataLoader(test_dataset, batch_size=config.batchSize, shuffle=False)

    # train(trainloader, config, testloader=testloader)

