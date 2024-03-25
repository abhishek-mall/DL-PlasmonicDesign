from torch.autograd import Variable
from tqdm import tqdm
import torch.utils.data as data_utils
import socket
from torch import nn, optim
import pdb
import numpy as np
from datetime import datetime
import os
import torch
from tensorboardX import SummaryWriter
import pandas as pd
import cel_model_exp
import time
special_info_for_run = "loss1.4 exp_wt 1e5 "
std_start_time = time.time()
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

def exp_loss(a,b):
    # print(a,b)
    diff = torch.abs(a-b)
    # print("_______________________________")
    # print(diff)
    # print("_______________________________")
    exp_diff = torch.exp(diff) -1
    sum_exp_diff = torch.sum(exp_diff)
    # print(sum_exp_diff)
    return sum_exp_diff

####################################
#Hyper-params
lr = 1e-3
exp_weightage = 1e5
mse_101_lambada = 1e5
smooth_lambada = 0
mse_101_gt4_lambada = 1e5
num_epochs = 10000
weight_decay=0.1
####################################


print('------------------------Data Pre-processing------------------------------')
test_data = (pd.read_csv("/home/Desktop/gan_in_physics/csv_square/squareTest_concated.csv", header=None))
test_df = pd.DataFrame(data=test_data)
test_tensor = torch.tensor(test_df.values)
train_data = (pd.read_csv("/home/Desktop/gan_in_physics/csv_square/squareTrain_concated.csv", header=None))
train_df = pd.DataFrame(data=train_data)
train_tensor = torch.tensor(train_df.values)
val_data = (pd.read_csv("/home/Desktop/gan_in_physics/csv_square/squareVal_concated.csv", header=None))
val_df = pd.DataFrame(data=val_data)
val_tensor = torch.tensor(val_df.values)

# print("type(test_tensor)",type(test_tensor))
print("test_tensor.shape",test_tensor.shape)
print("val_tensor.shape",val_tensor.shape)
print("train_tensor.shape",train_tensor.shape)

test_geometry = test_tensor[:,1:5]
val_geometry = val_tensor[:,1:5]
train_geometry = train_tensor[:,1:5]
print("test_geometry.shape",test_geometry.shape)
print("val_geometry.shape",val_geometry.shape)
print("train_geometry.shape",train_geometry.shape)

test_spectrum = test_tensor[:,5:]
val_spectrum = val_tensor[:,5:]
train_spectrum = train_tensor[:,5:]
print("test_spectrum.shape",test_spectrum.shape)
print("val_spectrum.shape",val_spectrum.shape)
print("train_spectrum.shape",train_spectrum.shape)
# print("test_spectrum",test_spectrum[:4,:5])
train = data_utils.TensorDataset(train_spectrum, train_geometry)
val = data_utils.TensorDataset(val_spectrum, val_geometry)
test = data_utils.TensorDataset(test_spectrum, test_geometry)

train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)
val_loader = data_utils.DataLoader(val, batch_size=10, shuffle=True)
test_loader = data_utils.DataLoader(test, batch_size=10, shuffle=False)
print('--------------------------------------------------------------')
log_dir = os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S') + ' ' +special_info_for_run)
writer = SummaryWriter(log_dir=log_dir)

 # for i, w in enumerate(model.parameters()):
 #            loss += scaling*l2_reg*torch.sum(w*w)/2.
 #            # loss += 0.1*torch.norm(w, 1)  ## adding L1 norm
 #            if i >= nb_set_parameters - 2:
 #                loss += l2_smooth*norm_diff(w)

def norm_diff(W):
    """ Special norm function for the last layer of the MTLR """
    dims=len(W.shape)
    if dims==1:
        diff = W[1:]-W[:-1]
    elif dims==2:
        diff = W[1:, :]-W[:-1, :]
    return torch.sum(diff*diff)

def train_model():
    enc_net = cel_model_exp.Encoder()
    dec_net = cel_model_exp.Decoder()
    # cel_net = cel_model_exp.CrossEntropyModel()

    enc_net.to(device)
    dec_net.to(device)
    # cel_net.to(device)
    # criterion_CEL = nn.CrossEntropyLoss()
    criterion_MSE = nn.MSELoss()

    # optimizer_cel = optim.Adam(cel_net.parameters(), lr=lr,weight_decay=weight_decay)
    optimizer_enc = optim.Adam(enc_net.parameters(), lr=lr,weight_decay=weight_decay)
    optimizer_dec = optim.Adam(dec_net.parameters(), lr=lr,weight_decay=weight_decay)
    
    criterion_MSE.to(device)
    # criterion_CEL.to(device)


    enc_net.train()
    dec_net.train()
    # cel_net.train()
    prev_mse_loss = 1e10

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0.0
        for spectrum, geometry in (train_loader):
            # print(spectrum.shape)
            # print(geometry.shape)

            spectrum = Variable(spectrum, requires_grad=True).to(device).float()
            geometry = Variable(geometry).to(device).float()
            cel_labels = (geometry[:,-1]/45).type(torch.LongTensor)

            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()
            # optimizer_cel.zero_grad()

            # cel_score = cel_net(spectrum)
            # probs = nn.Softmax(dim=1)(cel_score)
            # preds = torch.max(probs, 1)[1]

            att_4_mse = enc_net(spectrum)
            # att_4_mse[:,-1] = preds
            # print(att_4_mse[:,-1])
            # print(geometry[:,-1])
            # print(preds)
            # print(att_4_mse[:,-1])

            att_101_mse = dec_net(att_4_mse)
            att_101_mse_from_gt4 = dec_net(geometry)
            # att_101_mse_from_gt4 = torch.sigmoid(att_101_mse_from_gt4)
            # att_101_mse = torch.sigmoid(att_101_mse)

            # print(geometry[:,-1])
            # print(geometry[:,-1]/45)
            # print(type(geometry))
            # print(type(cel_labels))
            # print(cel_labels)
            # loss_CEL = criterion_CEL(cel_score, cel_labels)
            loss_MSE_4 = criterion_MSE(geometry, att_4_mse)
            loss_MSE_101 = criterion_MSE(spectrum,att_101_mse)
            loss_MSE_101_from_gt4 = criterion_MSE(spectrum,att_101_mse_from_gt4)

            # exp_loss_MSE_4 = exp_loss(geometry, att_4_mse)
            exp_loss_MSE_101 = exp_loss(spectrum,att_101_mse)
            exp_loss_MSE_101_from_gt4 = exp_loss(spectrum,att_101_mse_from_gt4)

            # exit()
            
            exp_total_loss =  mse_101_lambada*exp_loss_MSE_101 + mse_101_gt4_lambada*exp_loss_MSE_101_from_gt4
            loss = loss_MSE_4 + mse_101_lambada*loss_MSE_101 + mse_101_gt4_lambada*loss_MSE_101_from_gt4+exp_weightage*exp_total_loss

            # smooth_loss = norm_diff(att_101_mse) + norm_diff(att_101_mse_from_gt4)

            # att_101_mse[att_101_mse>1.0] = 1.0 - 1e-3
            # att_101_mse[att_101_mse<0.0] = 1e-3
            # att_101_mse_from_gt4[att_101_mse_from_gt4 >1.0] = 1.0 - 1e-3
            # att_101_mse_from_gt4[att_101_mse_from_gt4 <0.0] = 1e-3            

            # loss = smooth_lambada * smooth_loss+loss_MSE_4 + mse_101_lambada*loss_MSE_101 + mse_101_gt4_lambada*loss_MSE_101_from_gt4        
            # print("loss_MSE_4,loss_MSE_101:",loss_MSE_4.item(),loss_MSE_101.item())

            loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()
            # optimizer_cel.step()
            running_loss += loss.item()
            # running_corrects += torch.sum(preds == cel_labels.data)


        epoch_loss = running_loss / len(train_loader)
        # angle_cel_acc = running_corrects.double() / len(train_loader)
        # print("epoch_loss",epoch_loss)

        # writer.add_scalar('data/train_angle_cel_acc', angle_cel_acc, epoch)
        writer.add_scalar('data/train_total_loss', epoch_loss, epoch)
        # writer.add_scalar('data/train_smooth_loss', smooth_loss, epoch)
        writer.add_scalar('data/train_loss_MSE_4', loss_MSE_4, epoch)
        writer.add_scalar('data/train_loss_MSE_101', loss_MSE_101, epoch)
        writer.add_scalar('data/train_loss_MSE_101_from_gt4', loss_MSE_101_from_gt4, epoch)

        # writer.add_scalar('data/val_exp_loss_MSE_4', exp_loss_MSE_4, epoch)
        writer.add_scalar('data/val_exp_loss_MSE_101', exp_loss_MSE_101, epoch)
        writer.add_scalar('data/val_exp_loss_MSE_101_from_gt4', exp_loss_MSE_101_from_gt4, epoch)

        current_mse_losss = (loss_MSE_4 + loss_MSE_101_from_gt4)/2.0

        # exit()	
        if (epoch+1) % 100 ==0:
            print("[{}] Epoch: {}/{} Loss: {} MSELoss: {}".format("train", epoch+1, num_epochs, epoch_loss,current_mse_losss))
            test_model(enc_net,dec_net,criterion_MSE,epoch)
            print("time taken till now(hr:min)->", int((time.time() - std_start_time)/3600),':',int((time.time() - std_start_time)/60%60))
        # if (epoch+1)==num_epochs:
        if (prev_mse_loss > current_mse_losss):
            print("__________________prev_Loss: {} MSELoss: {}".format(prev_mse_loss,current_mse_losss))
            prev_mse_loss = current_mse_losss
            torch.save(enc_net, './save_model/{}_{}_enc.pt'.format(str(current_mse_losss.item())[:5],special_info_for_run))
            torch.save(dec_net, './save_model/{}_{}_dec.pt'.format(str(current_mse_losss.item())[:5],special_info_for_run))

            # torch.save(cel_net, './save_model/v2_cel_net.pt')




def test_model(enc_net,dec_net,criterion_MSE,epoch):
    enc_net.eval()
    dec_net.eval()
    running_loss = 0.0
    for spectrum, geometry in (val_loader):
        spectrum = Variable(spectrum, requires_grad=True).to(device).float()
        geometry = Variable(geometry).to(device).float()

        with torch.no_grad():
            att_4_mse = enc_net(spectrum)
            att_101_mse =(dec_net(att_4_mse))
            att_101_mse_from_gt4 = (dec_net(geometry))

        loss_MSE_4 = criterion_MSE(geometry, att_4_mse)
        loss_MSE_101 = criterion_MSE(spectrum,att_101_mse)
        loss_MSE_101_from_gt4 = criterion_MSE(spectrum,att_101_mse_from_gt4)

        # exp_loss_MSE_4 = exp_loss(geometry, att_4_mse)
        exp_loss_MSE_101 = exp_loss(spectrum,att_101_mse)
        exp_loss_MSE_101_from_gt4 = exp_loss(spectrum,att_101_mse_from_gt4)

        
        exp_total_loss =  mse_101_lambada*exp_loss_MSE_101 + mse_101_gt4_lambada*exp_loss_MSE_101_from_gt4
        loss = loss_MSE_4 + mse_101_lambada*loss_MSE_101 + mse_101_gt4_lambada*loss_MSE_101_from_gt4+exp_total_loss
        # print("loss_MSE_4,loss_MSE_101:",loss_MSE_4.item(),loss_MSE_101.item())

        running_loss += loss.item()

    epoch_loss = running_loss / len(val_loader)
    # print("epoch_loss",epoch_loss)

    writer.add_scalar('data/val_total_loss', epoch_loss, epoch)
    writer.add_scalar('data/val_loss_MSE_4', loss_MSE_4, epoch)
    writer.add_scalar('data/val_loss_MSE_101', loss_MSE_101, epoch)
    writer.add_scalar('data/val_loss_MSE_101_from_gt4', loss_MSE_101, epoch)

    # writer.add_scalar('data/val_exp_loss_MSE_4', exp_loss_MSE_4, epoch)
    writer.add_scalar('data/val_exp_loss_MSE_101', exp_loss_MSE_101, epoch)
    writer.add_scalar('data/val_exp_loss_MSE_101_from_gt4', exp_loss_MSE_101_from_gt4, epoch)

    print("[{}] Epoch: {}/{} Loss: {} ".format("val", epoch+1, num_epochs, epoch_loss))
    # print("time taken till now(hr:min)->", int((time.time() - std_start_time)/3600),':',int((time.time() - std_start_time)/60%60))




if __name__ == "__main__":
    train_model()   
    print("total_time_taken:",int(-(std_start_time - time.time())/3600)," hrs  ", int(-(std_start_time - time.time())/60%60), " mins")

