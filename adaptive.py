from pickletools import optimize
import torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants.analysis.rdp import compute_rdp,get_privacy_spent
from opacus.accountants.analysis import rdp as privacy_analysis
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import torch.nn.functional as F



class SampleConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"



class SampleConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SampleConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),#一般Kernel_size=5,padding=2
            nn.ReLU(),#standard activation fuction for cnn
            nn.MaxPool2d(kernel_size=2, stride=2))#demension_reduce
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out






def multi_skewness(grad_list):
    
    mean=torch.mean(grad_list,dim=0)
    # print(mean.size())
    # time.sleep(1000)
    # print(mean.unsqueeze(0))
    S=None

    for i in range(len(grad_list)):
        tt=grad_list[i].unsqueeze(0)
        if S==None:
            S=torch.mm(torch.transpose(tt-mean,dim0=1,dim1=0),tt-mean)
        else:
            S+=torch.mm(torch.transpose(tt-mean,dim0=1,dim1=0),tt-mean)
    S/=len(grad_list)
    S=torch.linalg.inv(S)
    res=None
    for i in range(len(grad_list)):
        ti=grad_list[i].unsqueeze(0)
        for j in range(len(grad_list)):
            tj=grad_list[j].unsqueeze(0)       
            b=torch.mm(ti-mean,S)
            b=torch.mm(b,torch.transpose(tj-mean,dim0=1,dim1=0))
            b=torch.pow(b,3)
            if res==None:
                res=b
            else:
                res+=b
            
    # print(res) 
    res/=len(grad_list)
    res/=len(grad_list)
    return res



def noise_max(norms,epsilon):
        
        noise=torch.distributions.laplace.Laplace(0,norms/epsilon).sample()
        return torch.max(norms)+0
    
    # def noise_max(self,mechanism,epsilon,percent):
    #     noise=torch.distributions.laplace.Laplace(0,self.max_grad_norm*percent/epsilon).sample()
    #     return torch.max(self.sample_norm_stack)*percent+noise

    
    
def percentile_clip(norms,epsilon,percentile):
    length=len(norms)
    tmp,_=torch.sort(norms,descending=False)
    index=int(length*percentile)
        # noise=torch.distributions.laplace.Laplace(0,self.max_grad_norm/epsilon).sample()
    noise=0
        
    return abs(tmp[index]+noise)



def opjective_func(args,clip_bound,norm_stack,step):
    # clip_bound=optimizer.max_grad_norm
    mins=10000000
    index=-1
    best_cb=-1
    file=open(args.save_path+"record"+str(args.clip_bound)+"_"+str(args.lr)+"/"+str(step)+".txt","w")
    for i in range(1,21):
        cb=(2*clip_bound)/20
        cb=cb*i
        bias=0
        cnt=256
        top=0
        
        for item in norm_stack:
            if top>cnt:
                break
            top+=1
            if item>cb:
                bias+=item-cb
                tmp=1-(cb/item)
                tmp=tmp*tmp
                bias+=item*item*tmp
                
        
        avg=torch.mean(norm_stack)
        
        bias/=cnt
        bias/=cnt
        bias*=avg
        
        
        var=(args.sigma*args.sigma*cb*cb*11181642)/(256*256)
        print("--------")
        print(avg)
        print(bias)
        print(var)
        expect=bias+var
        file.write(str(cb))
        file.write(" ")
        file.write(str(var))
        file.write(" ")
        file.write(str(bias))
        file.write(" ")
        file.write(str(expect))
        file.write("\n")
        if mins>expect:
            mins=expect
            index=i
            best_cb=cb
        # print(cb)
    #     print(expect)
    print("-----------")
    print(index)
    print(best_cb)
    file.close()
    return best_cb

def histgram_clip(args,norms,epsilon,percentile,count,bound,mode,step,save=False):
    stride=bound*2/count
    hist=[]
    for i in range(count):
        hist.append(0)
    for tmp in norms:
        index=int(tmp/stride)
        if index<count:
            hist[index]+=1
        else:
            hist[count-1]+=1
    target=len(norms)*percentile
    cnt=0
    
    if save==True:
        #save hist
        x_data=hist
        y_data=[i for i in range(count)]
        y_data=np.array(y_data)
        stride=torch.tensor(stride)
        print("______stride______")
        print(stride)
        length=stride.item()
        y_data=y_data*length
        tick=[i for i in range(count+1)]
        tick=np.array(tick)*length
        plt.bar(y_data+(length/2),x_data,length)
        plt.xticks(tick,rotation=90)
        if args.stage==-1:
            plt.savefig("./hist/histgram/"+str(args.percentile)+"/"+str(step)+".png")
        else:
            tmp=(args.stage-1)*4*2+step
            print("________save_step________")
            print(tmp)
            plt.savefig("./hist/histgram/"+str(args.percentile)+"/"+str(tmp)+".png")
        #save norm
        plt.clf()
        num_bins=0
        x_data=np.array(norms.cpu())
        bin_length=(max(x_data)-min(x_data))/40
        plt.hist(x_data,40)
        # plt.xticks(range(min(x_data),max(x_data)+bin_length,bin_length))
        if args.stage==-1:
            plt.savefig("./hist/norms/"+str(args.percentile)+"/"+str(step)+".png")
        else:
            tmp=(args.stage-1)*4*2+step
            plt.savefig("./hist/norms/"+str(args.percentile)+"/"+str(tmp)+".png")
        plt.clf()

    for i in range(count):
        if mode=="Laplace":
            noise=torch.distributions.laplace.Laplace(0,1/epsilon).sample()
        elif mode=="Gaussian":
            # print("Gaussian")
            noise=torch.normal(mean=0,std=torch.tensor(epsilon/1.0))
        else:
            print("no noise")
            noise=0
            
        hist[i]+=noise
        cnt+=hist[i]
        if cnt>target:
            return (stride*(i+1)+stride*(i))/2
            # return stride*(i+1)
    return  2*bound-bound/count

def count_under_threshold(self,threshold):
    count=0
    for tmp in self.origin_norm_stack:
        if tmp<threshold:
            count+=1
    return count

def mean_clip(self,epsilon):
    num=len(self.sample_norm_stack)
    noise=torch.distributions.laplace.Laplace(0,self.max_grad_norm/num*epsilon).sample()
    return torch.mean(self.sample_norm_stack)+noise

    
# def SVT_clip2(self,epsilon1,epsilon2,bound,threshold,sensitivity,C):
#     noise1=torch.distributions.laplace.Laplace(0,2/epsilon1).sample()
#     bound_list=[bound]*100
#     threshold+=noise1 #在阈值之下的sample数
#     count=0
#     i=0
#     res=[]
#     for bound in bound_list:
#         tmp=self.count_under_threshold(bound)
#         noise2=torch.distributions.laplace.Laplace(0,4/epsilon1).sample()
#         if tmp+noise2>threshold:
#             res.append(i)
#             count+=1
#         if count==C:
#             return res
#         noise1=torch.distributions.laplace.Laplace(0,2/epsilon1).sample()
#         threshold+=noise1

class adaptive_control():
    DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    resample_steps=0
    rdp=[]
    def compute(self,q,noise_multiplier):
        rdp=compute_rdp(q=q,noise_multiplier=noise_multiplier,steps=self.resample_steps,orders=self.DEFAULT_ALPHAS)
        return rdp
        

    