import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F


def train(model, 
          dataloader ,
          device,
          output_dir,
          optimizer = optim.Adam(model.parameters()),
          loss_fn = nn.MSELoss(),
          input_feature : int = 1,
          mode = 'normal', 
          epochs : int = 60000, 
          checkpoint : int = 300,
          ):
    
    assert mode == 'normal' or 'enc_dec' , "Choose 'normal', or 'enc_dec' if have encoders and decoders "

    optimizer = optimizer
    loss_fn = loss_fn
    train_ins_error=[]

    for epoch in tqdm(range(epochs)):
        model.train()

        if mode == 'normal':   ## One input sequence and one output sequence
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if input_feature == 1:
                    X_batch, y_batch = X_batch.unsqueeze(-1),y_batch.unsqueeze(-1)
                y_pred = model(X_batch)        
                loss = loss_fn(y_pred, y_batch )     
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_ins_error.append(loss.item())

        else:
            for src, tgt , y in dataloader:
                src, tgt , y = src.to(device), tgt.to(device), y.to(device)
                if input_feature == 1:
                    src, tgt, y = src.unsqueeze(-1),tgt.unsqueeze(-1),y.unsqueeze(-1)
                y_pred = model(src = src, tgt = tgt)    
                loss = loss_fn(y_pred, y )   
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_ins_error.append(loss.item())

        if (epoch+1) % checkpoint == 0:

            # show training loss

            temp_error = np.stack(train_ins_error)
            fig, ax = plt.subplots(figsize=(10,8))
            lw=1
            ax.plot(temp_error,color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='Training error')
            ax.set_yscale('log')
            leg = ax.legend(loc='upper right',frameon = True)
            leg.get_frame().set_edgecolor('black')
            fig.savefig(output_dir+'/train_error_epoch{:d}.png'.format(epoch),bbox_inches='tight' )

            # save model

            error1 = np.stack(train_ins_error)
            error1 = torch.Tensor(error1)
            torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': error1.detach().cpu(),
                }, output_dir+"/checkpoint_{:d}.pt".format(epoch))
            

        #  roll out the results 

        model.eval()
              
        X_train = X_train.to(device)
        temp = X_train[0].unsqueeze(0)
        y_pred_test = temp.clone()
        with torch.no_grad():
            for i in range(train_size+test_size+1):
                temp = model(temp)
                y_pred_test = torch.cat((y_pred_test, temp[:,-1,:].unsqueeze(1)), dim=1)
                temp = y_pred_test[:,-lookback:,:]

        # y_pred_test = data_normalizer.denormalize(y_pred_test)

        print(xx.shape, y_pred_test.shape)
        # plot out the distribution 
        fig, ax = plt.subplots(figsize=(100,8))
        lw=3
        ax.plot(xx, signal,color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='Signal')
        # ax.plot(xx[:lookback], y_pred_test.squeeze().detach().cpu().numpy()[:lookback],color = 'C0',linestyle='dashed',linewidth=lw,alpha=1,label='Output')
        ax.plot(xx[lookback:train_size], y_pred_test.squeeze().detach().cpu().numpy()[lookback:train_size],color = 'C1',linestyle='dashed',linewidth=lw,alpha=1,label='Fit')
        ax.plot(xx[train_size-1:], y_pred_test.squeeze().detach().cpu().numpy()[train_size-1:],color = 'C2',linestyle='dashed',linewidth=lw,alpha=1,label='Prediction')

        ax.set_xlabel('time')
        ax.set_ylabel('output')
        leg = ax.legend(loc='lower right',frameon = True )
        leg.get_frame().set_edgecolor('black')
        # fig.show()
        fig.savefig(output_dir+'/prediction_epcoh{:d}.png'.format(epoch),bbox_inches='tight' )
    tqdm.write('{:e}'.format(loss))
    