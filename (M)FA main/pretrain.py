import os
import time
import torch
import argparse
from torch import nn
from torch.utils import data
from torchvision import transforms
from tools.utils import *
import tools.model as models
from dataset1.scene_dataset import *


def main(args):
    if args.dataID==1:
        DataName = 'UCM'
        num_classes = 21
        classname = ('agricultural','airplane','baseballdiamond',
                        'beach','buildings','chaparral',
                        'denseresidential','forest','freeway',
                        'golfcourse','harbor','intersection',
                        'mediumresidential','mobilehomepark','overpass',
                        'parkinglot','river','runway',
                        'sparseresidential','storagetanks','tenniscourt')

    elif args.dataID==2:        
        DataName = 'AID'
        num_classes = 30
        classname = ('airport','bareland','baseballfield',
                        'beach','bridge','center',
                        'church','commercial','denseresidential',
                        'desert','farmland','forest',
                        'industrial','meadow','mediumresidential',
                        'mountain','parking','park',
                        'playground','pond','port',
                        'railwaystation','resort','river',
                        'school','sparseresidential','square',
                        'stadium','storagetanks','viaduct')
    
                  
    print_per_batches = args.print_per_batches
    save_path_prefix = args\
                           .save_path_prefix+DataName+'/Pretrain/'+args.network+'/'
    
    if os.path.exists(save_path_prefix)==False:
        os.makedirs(save_path_prefix)

    composed_transforms = transforms.Compose([
            transforms.Resize(size=(args.crop_size,args.crop_size)),            
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    train_loader = data.DataLoader(
        scene_dataset(root_dir=args.root_dir,pathfile='./dataset1/'+DataName+'_train.txt', transform=composed_transforms),
        batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = data.DataLoader(
        scene_dataset(root_dir=args.root_dir,pathfile='./dataset1/'+DataName+'_test.txt', transform=composed_transforms),
        batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    ###################Network Definition###################
    if args.network=='alexnet':
        Model = models.alexnet(pretrained=True)
        Model.classifier._modules['6'] = nn.Linear(4096, num_classes) 
    elif args.network=='vgg11':
        Model = models.vgg11(pretrained=True)  
        Model.classifier._modules['6'] = nn.Linear(4096, num_classes)
    elif args.network=='vgg16':
        Model = models.vgg16(pretrained=True)  
        Model.classifier._modules['6'] = nn.Linear(4096, num_classes)
    elif args.network=='vgg19':
        Model = models.vgg19(pretrained=True)  
        Model.classifier._modules['6'] = nn.Linear(4096, num_classes)
    elif args.network=='inception':
        Model = models.inception_v3(pretrained=True, aux_logits=False)  
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network=='resnet18':
        Model = models.resnet18(pretrained=True)  
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network=='resnet50':
        Model = models.resnet50(pretrained=True)  
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network=='resnet101':
        Model = models.resnet101(pretrained=True)  
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network=='resnext50_32x4d':
        Model = models.resnext50_32x4d(pretrained=True)  
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network=='resnext101_32x8d':
        Model = models.resnext101_32x8d(pretrained=True)  
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network=='densenet121':
        Model = models.densenet121(pretrained=True)  
        Model.classifier = nn.Linear(1024, num_classes)
    elif args.network=='densenet169':
        Model = models.densenet169(pretrained=True)  
        Model.classifier = nn.Linear(1664, num_classes)
    elif args.network=='densenet201':
        Model = models.densenet201(pretrained=True)  
        Model.classifier = nn.Linear(1920, num_classes)
    elif args.network=='regnet_x_400mf':
        Model = models.regnet_x_400mf(pretrained=True)  
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network=='regnet_x_8gf':
        Model = models.regnet_x_8gf(pretrained=True)  
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif args.network=='regnet_x_16gf':
        Model = models.regnet_x_16gf(pretrained=True)  
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)

    Model = torch.nn.DataParallel(Model).cuda()
    Model_optimizer = torch.optim.Adam(Model.parameters(),lr=args.lr)
    num_batches = len(train_loader)
    
    cls_loss = torch.nn.CrossEntropyLoss()
    num_steps = args.num_epochs*num_batches
    hist = np.zeros((num_steps,3))
    index_i = -1
    
    for epoch in range(args.num_epochs):
        for batch_index, src_data in enumerate(train_loader):
            index_i += 1
    
            tem_time = time.time()
            Model.train()
            Model_optimizer.zero_grad()

            X_train, Y_train, _ = src_data
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()
            
            _,output = Model(X_train)

            # CE Loss            
            _, src_prd_label = torch.max(output, 1)            
            cls_loss_value = cls_loss(output, Y_train)
            cls_loss_value.backward()

            Model_optimizer.step()            
            
            hist[index_i,0] = time.time()-tem_time
            hist[index_i,1] = cls_loss_value.item()   
            hist[index_i,2] = torch.mean((src_prd_label == Y_train).float()).item() 

            tem_time = time.time()
            if (batch_index+1) % print_per_batches == 0:
                print('Epoch %d/%d:  %d/%d Time: %.2f cls_loss = %.3f acc = %.3f \n'\
                %(epoch+1, args.num_epochs,batch_index+1,num_batches,
                np.mean(hist[index_i-print_per_batches+1:index_i+1,0]),
                np.mean(hist[index_i-print_per_batches+1:index_i+1,1]),
                np.mean(hist[index_i-print_per_batches+1:index_i+1,2])))
        


    OA_new,_ = test_acc(Model,classname, val_loader, epoch+1,num_classes,print_per_batches=10)
    
    model_name = 'epoch_'+str(epoch+1)+'_OA_'+repr(int(OA_new*10000))+'.pth'

    print('Save Model') 
    torch.save(Model.state_dict(), os.path.join(save_path_prefix, model_name))
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--dataID', type=int, default=2)
   # parser.add_argument('--network', type=str, default='resnet18',
    parser.add_argument('--network', type=str, default='regnet_x_400mf',
                        help='alexnet,vgg11,vgg16,vgg19,inception,resnet18,resnet50,resnet101,resnext50_32x4d,resnext101_32x8d,densenet121,densenet169,densenet201,regnet_x_400mf,regnet_x_8gf,regnet_x_16gf')
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--root_dir', type=str, default='',help='dataset path.')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--print_per_batches', type=int, default=5)

    main(parser.parse_args())
