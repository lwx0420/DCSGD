import argparse
from distutils.command.build import build
from email.policy import default
import imp
from pickletools import optimize
import random
from re import S, T
from statistics import mode
import string
import time
from tkinter import N
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.utils.data
import opacus
from tqdm import tqdm
import numpy as np
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.optimizers.utils import params
import adaptive
from opacus.accountants.analysis import rdp as privacy_analysis
from scipy import stats
from opacus.accountants.analysis.rdp import compute_rdp,get_privacy_spent
from opacus.data_loader import DPDataLoader

def _get_flat_grad_sample(p: torch.Tensor):

    if not hasattr(p, "grad_sample"):
        raise ValueError(
            "Per sample gradient not found. Are you using GradSampleModule?"
        )
    if isinstance(p.grad_sample, torch.Tensor):
        return p.grad_sample
    elif isinstance(p.grad_sample, list):
        return torch.cat(p.grad_sample, dim=0)
    else:
        raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def accuracy(preds, labels):
    return (preds == labels).mean()




def build_hist(args,model,resample_dataloader,optimizer,device):
    criterion = torch.nn.CrossEntropyLoss()
    # model.train()
    optimizer.zero_grad()
    model.zero_grad()
    norm_stack=torch.tensor([])
    norm_stack=norm_stack.to(device)
    sum_grad=None
    sum_clip_grad=[]
    dimension=0
    for i in range(1,21):
        sum_clip_grad.append(None)
    
    clip_bound=optimizer.max_grad_norm
    norm_cnt=0
    split_dataloader=[]
    physical_batch=64
    
    for image,target in resample_dataloader:
        sp1=torch.chunk(input=image,chunks=physical_batch,dim=0)
        sp2=torch.chunk(input=target,chunks=physical_batch,dim=0)
        print(image.size())
        for i in range(0,len(sp1)):
            split_dataloader.append([sp1[i],sp2[i]])
        break


    for sample in split_dataloader:
        image=sample[0]
        if args.model_type=="linear":
            image=image.reshape(-1,28*28)
        target=sample[1]
        image = image.to(device)
        target = target.to(device)
        output=model(image)
        loss=criterion(output,target)
        loss.backward()    
        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in optimizer.grad_samples
        ]
        
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        pp=per_sample_norms
        norm_stack=torch.cat([norm_stack,pp],dim=0)
        norm_cnt+=len(per_sample_norms)

        grad_list=None
        for p in optimizer.params:
            num=_get_flat_grad_sample(p).size()[0]
            break
        for p in optimizer.params:
            grad_sample = _get_flat_grad_sample(p)
            grad_sample=grad_sample.reshape(num,-1)
            if grad_list==None:
                grad_list=grad_sample
            else:
                grad_list=torch.cat([grad_list,grad_sample],dim=1)
        dimension=grad_list.size()[1]   
        
        optimizer.zero_grad()
        model.zero_grad()
              
    optimizer.control.resample_steps+=1

    stride=400/20
    hist=[]
    for i in range(20):
        hist.append(0)
    for tmp in norm_stack:
        if int(tmp/stride)>19:
            hist[19]+=1
        else:
            hist[int(tmp/stride)]+=1
    for i in range(20):
        noise=torch.normal(mean=0,std=torch.tensor(args.sigma_t/1.0))
        hist[i]+=noise
    # initialrange=50
    # while(True):
    #     stride=initialrange/20
    #     hist=[]
    #     for i in range(20):
    #         hist.append(0)
    #     for tmp in norm_stack:
    #         if int(tmp/stride)>19:
    #             hist[19]+=1
    #         else:
    #             hist[int(tmp/stride)]+=1
    #     for i in range(20):
    #         noise=torch.normal(mean=0,std=torch.tensor(args.sigma_t/1.0))
    #         hist[i]+=noise
    #     if(hist[19]>2*stride):
    #         initialrange=2*initialrange
    #         continue
    #     tmpsum=0
    #     for i in range(10,20):
    #         tmpsum=tmpsum+hist[i]
    #     if tmpsum<stride:
    #         initialrange=0.5*initialrange
    #         continue
    #     break
    print("Dimension")
    print(dimension)
    return hist, dimension, len(norm_stack)
    
    

def resample(args,optimizer,hist,lennorm,dimension):
    
    # dimension=dimension
    clip_bound=optimizer.max_grad_norm
    stride=400/20
    mins=100000000000
    index=-1
    best_cb=-1
    file=open(args.save_path+"record"+str(args.clip_bound)+"_"+str(args.lr)+"/"+str(optimizer.control.resample_steps)+".txt","w")
    
    for i in range(1,21):
        sum_norm=0
        cb=(clip_bound/10)*i
        var=(args.sigma*args.sigma*cb*cb*dimension)/(args.batchsize_train*args.batchsize_train)
        # var*=(optimizer.state_dict()['param_groups'][0]['lr'])
        
        
        bias=0
        for i in range(20):
            mid=stride/2
            mid+=stride*i
            if mid>cb:
                bias+=hist[i]*(mid-cb)
        bias*=bias
        bias/=lennorm
        bias/=lennorm
        
        
        expect=bias+var
        expect=bias+var
        file.write(str(cb))
        file.write(" ")
        file.write(str(var))
        file.write(" ")
        file.write(str(bias))
        file.write(" ")
        file.write(str(expect))
        file.write(" ")
        file.write("\n")
            
        if mins>expect:
            mins=expect
            index=i
            best_cb=cb
    print("-----------")
    print(index)
    print(best_cb)
    file.close()
    optimizer.max_grad_norm=best_cb
    return optimizer

def test(args,model,test_loader,optimizer,privacy_engine,epoch,device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images,target in tqdm(test_loader):
            if args.model_type=="linear":
                images=images.reshape(-1,28*28)
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc1)

        top1_avg = np.mean(top1_acc)
        print(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} ")
        return top1_avg




def train(args, model, train_loader,resample_dataloader ,optimizer, privacy_engine, epoch, device, resample_num, project_gaussian,project2,loss_rec,mean_norm_rec):
    
    model.train()
    criterion=torch.nn.CrossEntropyLoss()
    losses=[]
    top_acc=[]
    model.zero_grad()
    optimizer.zero_grad()
    cnt=0
    resample_num=args.resample_num
    trains=2
    hist,dimension,lennorm=build_hist(args,model,resample_dataloader,optimizer,device)
    while True:
        mgn=optimizer.max_grad_norm
        optimizer=resample(args,optimizer,hist,lennorm,dimension)
        print("-----")
        print(mgn)
        print(optimizer.max_grad_norm)
        #To avoid the error caused by float precision, use inequality instead of "=="
        if optimizer.max_grad_norm<=0.05*mgn or optimizer.max_grad_norm>=1.95*mgn:
            continue
        break
    optimizer.norm_record.append(optimizer.max_grad_norm)
    # mean_norm_rec.append(optimizer.mean_norm)
    trains-=1
    hist=[]
    
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=128, 
        optimizer=optimizer
    ) as memory_safe_data_loader:

        for i,(images,target) in enumerate(tqdm(memory_safe_data_loader)):
            
            if args.model_type=="linear":
                images=images.reshape(-1,28*28)
            images=images.to(device)
            target=target.to(device)

            cnt+=len(images)
            
            output=model(images)
            loss=criterion(output,target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc=accuracy(preds,labels)
            top_acc.append(acc)

            losses.append(loss.item())

            loss.backward()
            optimizer.batch_acc.append(acc)
            # optimizer.zero_grad()
            optimizer.step()
            optimizer.zero_grad()
            
            if args.dp_able and i!=0 and i!=1 and args.adaptive==1:
                alpha=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
                qs=privacy_engine.accountant.steps[0][1]
                sig=privacy_engine.accountant.steps[0][0]
                ste=privacy_engine.accountant.steps[0][2]
                rdp1=compute_rdp(q=qs,noise_multiplier=sig,steps=ste+1,orders=alpha)
                eps, best_alpha1 = privacy_analysis.get_privacy_spent(
                    orders=optimizer.control.DEFAULT_ALPHAS, rdp=rdp1, delta=args.delta
                )
                rdp=optimizer.control.compute(1/len(resample_dataloader),args.sigma_t)

                total=rdp1+rdp
                eps_total, best_alpha2 = privacy_analysis.get_privacy_spent(
                    orders=optimizer.control.DEFAULT_ALPHAS, rdp=total, delta=args.delta
                )


                optimizer.total_eps=eps_total
                
                # judge if excess the target epsilon
                if eps_total>=args.target_eps:
                    print("total_eps_break:")
                    print(eps_total)
                    print(eps)
                    return

            if cnt>resample_num and args.adaptive==1 and trains!=0:
                hist,dimension,lennorm=build_hist(args,model,resample_dataloader,optimizer,device)
                mean_norm_rec.append(optimizer.mean_norm)
                loss_rec.append(loss.item())
                optimizer.resample_acc.append(acc)
                while True:
                    mgn=optimizer.max_grad_norm
                    optimizer=resample(args,optimizer,hist,lennorm,dimension)
                    if optimizer.max_grad_norm<=0.05*mgn or optimizer.max_grad_norm>=1.95*mgn:
                        continue
                    break
                hist=[]
                optimizer.norm_record.append(optimizer.max_grad_norm)
                trains-=1
                cnt=0

            if i%args.print_frequence==0 and i!=0:
                
                if args.dp_able:
                    alpha=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

                    rdp1=privacy_engine.accountant.get_rdp(delta=args.delta,
                        alphas=alpha,)
                    epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(
                        delta=args.delta,
                        alphas=alpha,
                    )
                    if args.adaptive==1:
                        

                        rdp=optimizer.control.compute(1/len(resample_dataloader),args.sigma_t)
                        eps, best_alpha1 = privacy_analysis.get_privacy_spent(
                            orders=optimizer.control.DEFAULT_ALPHAS, rdp=rdp, delta=args.delta
                        )
                        total=rdp1+rdp
                        eps_total, best_alpha2 = privacy_analysis.get_privacy_spent(
                            orders=optimizer.control.DEFAULT_ALPHAS, rdp=total, delta=args.delta
                        )
                        print("total")
                        
                        print(eps_total)
                        print(epsilon)
                        print(eps)
                    max_grad_norm=optimizer.max_grad_norm
                    print(
                        f"Train Epoch: {epoch} "
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top_acc):.6f} "
                        f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha} "
                        f"max_grad_norm: {max_grad_norm} "
                        f"lr: {optimizer.state_dict()['param_groups'][0]['lr']} "
                        f"re_step: {optimizer.control.resample_steps} "
                    )

                else:
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top_acc):.6f} "
                        f"lr: {optimizer.state_dict()['param_groups'][0]['lr']} "
                    )
    
    if args.dp_able:
        alpha=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

        rdp1=privacy_engine.accountant.get_rdp(delta=args.delta,
            alphas=alpha,)
        epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(
            delta=args.delta,
            alphas=alpha,
        )

        rdp=optimizer.control.compute(1/len(resample_dataloader),args.sigma_t)
        eps, best_alpha1 = privacy_analysis.get_privacy_spent(
            orders=optimizer.control.DEFAULT_ALPHAS, rdp=rdp, delta=args.delta
        )
        total=rdp1+rdp
        eps_total, best_alpha2 = privacy_analysis.get_privacy_spent(
            orders=optimizer.control.DEFAULT_ALPHAS, rdp=total, delta=args.delta
        )
        print("total")
        
        print(eps_total)
        print(epsilon)
        print(eps)
        max_grad_norm=optimizer.max_grad_norm
        print(
            f"Train Epoch: {epoch} "
            f"Loss: {np.mean(losses):.6f} "
            f"Acc@1: {np.mean(top_acc):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha} "
            f"max_grad_norm: {max_grad_norm} "
            f"lr: {optimizer.state_dict()['param_groups'][0]['lr']} "
            f"re_step: {optimizer.control.resample_steps} "
        )

def main():
    args=parse_args()
    print(args)
    device=args.device
    # set_seed(args.seed)
    if args.adjust_freq<200:
        raise NotImplementedError(
            "adjust_freq is too small"
        )
        return 
    # args.dp_able=False

    if args.dataset=="CIFAR10":
        augmentations = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        train_transform = transforms.Compose(
            augmentations + normalize if not args.dp_able else normalize
        )
        test_transform = transforms.Compose(normalize)

        train_dataset=torchvision.datasets.CIFAR10(root=args.data_path,transform=train_transform,download=False,train=True)
        test_dataset=torchvision.datasets.CIFAR10(root=args.data_path,transform=test_transform,download=False,train=False)
        args.delta=1/len(train_dataset)
        
    
        train_dataloader=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchsize_train,
            pin_memory=True,
        )

        test_dataloader=torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batchsize_test,
        )
        resample_dataloader=None
        if args.adaptive==1:
            resample_dataloader=torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.resample_batchsize,
                pin_memory=True,
                shuffle=True,
            )
            resample_dataloader=DPDataLoader.from_data_loader(
                resample_dataloader
            )
            
    elif args.dataset=="MNIST":
        
        transform=transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
        ])
        if args.model_type=="linear" or args.model_type=="small":
            transform=transforms.Compose([
                # transforms.Resize((224, 224)),
                # transforms.Grayscale(3),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
            ])
                
        train_dataset=torchvision.datasets.MNIST(root=args.data_path,transform=transform,download=True,train=True)
        test_dataset=torchvision.datasets.MNIST(root=args.data_path,transform=transform,download=True,train=False)
        args.delta=1/len(train_dataset)
        print("____MNIST______")
        print(len(train_dataset))
        
    
        train_dataloader=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchsize_train,
            pin_memory=True,
        )

        test_dataloader=torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batchsize_test,
        )
        resample_dataloader=None
        if args.adaptive==1:
            resample_dataloader=torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.resample_batchsize,
                pin_memory=True,
                shuffle=True,
            )
            resample_dataloader=DPDataLoader.from_data_loader(
                resample_dataloader
            )
    elif args.dataset=="SVHN":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        train_dataset=torchvision.datasets.SVHN(root=args.data_path,transform=transform,download=True,split="train")
        test_dataset=torchvision.datasets.SVHN(root=args.data_path,transform=transform,download=True,split="test")
        args.delta=1/len(train_dataset)
        print("____SVHN______")
        print(len(train_dataset))
        
    
        train_dataloader=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchsize_train,
            pin_memory=True,
        )

        test_dataloader=torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batchsize_test,
        )
        resample_dataloader=None
        if args.adaptive==1:
            resample_dataloader=torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.resample_batchsize,
                pin_memory=True,
                shuffle=True,
            )
            resample_dataloader=DPDataLoader.from_data_loader(
                resample_dataloader
            )

    else:
        raise NotImplementedError(
            "there is no such dataset"
        )
    
    if args.model_type=="resnet":
        if args.dataset=="SVHN":
            n_classes=10
        else:
            n_classes=len(train_dataset.classes)
        model=torchvision.models.resnet18(num_classes=n_classes)
        model=model.to(device)
    elif args.model_type=="resnet34":
        print("res34")
        if args.dataset=="SVHN":
            n_classes=10
        else:
            n_classes=len(train_dataset.classes)
        model=torchvision.models.resnet34(num_classes=n_classes,pretrained=False)
        # torchvision.models.
        model=model.to(device)
    elif args.model_type=="MNIST":
        n_classes=len(train_dataset.classes)
        model=torchvision.models.resnet18(num_classes=n_classes)
        model=model.to(device)
    elif args.model_type=="small":
        model=adaptive.SampleConvNet()
        model=model.to(device)
        print("small model")
    elif args.model_type=="linear":
        print("linear")
        model=torch.nn.Linear(28*28,10)
        model=model.to(device)
       
    else:
        raise NotImplementedError(
            "there is no such model"
        )
    
    

    if args.optim=="SGD":
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9
        )
    elif args.optim=="Adam":
        optimizer=torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )
    else:
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
        )


    
    #optimizer必须在model完全处理完后添加（比如fix完后）
    privacy_engine=None
    if args.dp_able:
        privacy_engine=opacus.PrivacyEngine()
        model = ModuleValidator.fix(model)
        model=model.to(device)
        if args.optim=="SGD":
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=0.9,
            )
        elif args.optim=="Adam":
            print("adam-----------")
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
            )
        else:
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
            )
        model,optimizer,train_dataloader=privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            noise_multiplier=args.sigma,
            max_grad_norm=args.clip_bound,
        )
        # model,optimizer,train_dataloader=privacy_engine.make_private_with_epsilon(
        #     module=model,
        #     optimizer=optimizer,
        #     data_loader=train_dataloader,
        #     epochs=args.epochs,
        #     target_delta=args.delta,
        #     target_epsilon=8,
        #     max_grad_norm=args.clip_bound,
        # )
        optimizer.control=adaptive.adaptive_control()
        optimizer.batch_acc=[]
        optimizer.resample_acc=[]
        optimizer.norm_record=[]
        optimizer.mean_norm=0
        optimizer.total_eps=0


        
    # project_gaussian=torch
    project_gaussian=torch.randn(1,2).to(args.device)
    project2=torch.randn(1,1).to(args.device)
    acc=[]
    import os
    os.system("mkdir "+args.save_path+"record"+str(args.clip_bound)+"_"+str(args.lr))
    loss_rec=[]
    mean_norm_rec=[]
    for epochs in range(1,args.epochs+1):
        # if epochs==1 and args.adaptive==1: optimizer=resample(args,model,resample_dataloader,optimizer,device,resample_num=args.resample_num)  
        train(args,model,train_dataloader,resample_dataloader,optimizer,privacy_engine,epochs,device,resample_num=args.resample_num,project_gaussian=project_gaussian,project2=project2,loss_rec=loss_rec,mean_norm_rec=mean_norm_rec)
        tmp=test(args,model,test_dataloader,optimizer,privacy_engine,epochs,device)
        acc.append(tmp)
    

    yaxis=torch.tensor(optimizer.norm_record)
    plt.plot(yaxis)
    plt.title("optimizer="+args.optim+"  clip_bound="+str(args.clip_bound))
    plt.xlabel("iteration")
    plt.ylabel("norm")
    if args.adaptive==1:
        plt.savefig(args.save_path+args.optim+"_"+str(args.clip_bound)+"_"+str(args.lr)+"_"+str(args.resample_num)+"_"+str(args.percentile)+"_norm_eps"+str(optimizer.total_eps)+".png")
        file=open(args.save_path+args.optim+"_"+str(args.clip_bound)+"_"+str(args.lr)+"_"+str(args.resample_num)+"_"+str(args.percentile)+"_norm.txt","w")
        file.write(str(optimizer.norm_record))
        file=open(args.save_path+args.optim+"_"+str(args.clip_bound)+"_"+str(args.lr)+"_"+str(args.resample_num)+"_"+str(args.percentile)+"_meannorm.txt","w")
        file.write(str(mean_norm_rec))
        file=open(args.save_path+args.optim+"_"+str(args.clip_bound)+"_"+str(args.lr)+"_"+str(args.resample_num)+"_"+str(args.percentile)+"_loss.txt","w")
        file.write(str(loss_rec))
        # file=open(args.save_path+args.optim+"_"+str(args.clip_bound)+"_"+str(args.lr)+"_"+str(args.resample_num)+"_"+str(args.percentile)+"_resample_acc.txt","w")
        # file.write(str(optimizer.resample_acc))

    #else:
        #plt.savefig(args.save_path+args.optim+"_"+str(args.clip_bound)+"_"+str(args.lr)+"_norm.png")

    
    plt.clf()
    yaxis=acc
    plt.plot(yaxis)
    plt.title("optimizer="+args.optim+"  clip_bound="+str(args.clip_bound))
    plt.xlabel("epoch")
    plt.ylabel("acc")
    file=open(args.save_path+"final_res.txt","a")
    file.write(str(format(acc[args.epochs-1],'.4f')))
    file.write(" ")
    file.close()
    if args.adaptive==1:
        plt.savefig(args.save_path+args.optim+"_"+str(args.clip_bound)+"_"+str(args.lr)+"_"+str(args.resample_num)+"_"+str(args.percentile)+"_"+str(format(acc[args.epochs-1],'.4f'))+"_acc.png")
        
    else:
        plt.savefig(args.save_path+args.optim+"_"+str(args.clip_bound)+"_"+str(args.lr)+"_"+str(format(acc[args.epochs-1],'.4f'))+"_acc.png")
    

    plt.clf()
    

    


def parse_args():
    parser=argparse.ArgumentParser(description="adaptive dp")
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batchsize-test",
        default=256,
        type=int,
        help="batch size of test dataset",
    )
    parser.add_argument(
        "--batchsize-train",
        default=32,
        type=int,
        help="batch size of training dataset",
    )
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--seed",
        default=101,
        type=int,
        help="the random seed",
    )
    parser.add_argument(
        "--model_type",
        default="resnet",
        type=str,
        help="model type",
    )
    parser.add_argument(
        "--dataset",
        default="CIFAR10",
        type=str,
        help="dataset",
    )
    parser.add_argument(
        "--dp_able",
        default=False,
        type=bool,
        help="if using DP method to train",
    )
    parser.add_argument(
        "--sigma",
        default=0.3,
        type=float,
        help="noise multiplier",
    )
   
    parser.add_argument(
        "--data_path",
        type=str,
        help="the path of dataset",
    )
    parser.add_argument(
        "--log-dir",
        default="./tmp/log",
        type=str,
        help="the path of log"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
    )
    parser.add_argument(
        "--clip_bound",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--optim",
        default="SGD",
        help="the optimizer used"
    )
    parser.add_argument(
        "--resample_batchsize",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--print_frequence",
        default=50,
    )
    parser.add_argument(
        "--DP_mode",
        default="oringinal",
        help="decide the DP_mode, [oringinal,noisy_max,noisy_percent,noisy_decay,histogram,SVT,]",
    )
    parser.add_argument(
        "--delta",
        default=1/50000,
    )
    parser.add_argument(
        "--adaptive",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--percentile",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--resample_num",
        default=2000,
        type=int,
    )
    parser.add_argument(
        "-adjust_freq",
        default=2000,
        type=int,
    )
    parser.add_argument(
        "--save_path",
        default="./",
        type=str,
    )
    parser.add_argument(
        "--stage",
        default="-1",
        type=int,
    )
    parser.add_argument(
        "--pretrain",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--target_eps",
        default=2.0,
        type=float,
    )
    parser.add_argument(
        "--sigma_t",
        default=2.0,
        type=float,
    )
    parser.add_argument(
        "--bins",
        default=20,
        type=int,
    )
    return parser.parse_args()


if __name__=="__main__":
    main()
