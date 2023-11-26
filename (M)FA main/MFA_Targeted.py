import os
import argparse
from tools.utils import *
from torch.nn import functional as F
import tools.model as models
from dataset1.scene_dataset import *
from torch import nn
from torch.utils import data
from tqdm import tqdm


def main(args):
    if args.dataID == 1:
        DataName = 'UCM'
        num_classes = 21
        if args.attack_func[:2] == 'mi':
            mix_file = 'H:/code/UAE-RS-main/dataset1/UCM_' + args.attack_func + '_sample.tif'
    elif args.dataID == 2:
        DataName = 'AID'
        num_classes = 30
        if args.attack_func[:2] == 'mi':
            mix_file = 'H:/code/UAE-RS-main/dataset1/AID_' + args.attack_func + '_sample.jpg'

    save_path_prefix = args.save_path_prefix + DataName + '_adv/' + args.attack_func + '/' + args.surrogate_model + '/'

    if os.path.exists(save_path_prefix) == False:
        os.makedirs(save_path_prefix)

    composed_transforms = transforms.Compose([
        transforms.Resize(size=(args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    imloader = data.DataLoader(
        scene_dataset(root_dir=args.root_dir, pathfile='H:/code/UAE-RS-main/dataset1/' + DataName + '_test.txt',
                      transform=composed_transforms),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    ###################Surrogate Model Definition###################
    args.surrogate_model1 = 'alexnet'
    surrogate_model1 = models.alexnet(pretrained=False)
    surrogate_model1.classifier._modules['6'] = nn.Linear(4096, num_classes)
    args.surrogate_model2 = 'resnet18'
    surrogate_model2 = models.resnet18(pretrained=False)
    surrogate_model2.fc = torch.nn.Linear(surrogate_model2.fc.in_features, num_classes)
    args.surrogate_model3 = 'densenet121'
    surrogate_model3 = models.densenet121(pretrained=False)
    surrogate_model3.classifier = nn.Linear(1024, num_classes)
    # args.surrogate_model4 = 'regnet_x_400mf'
    # surrogate_model4 = models.regnet_x_400mf(pretrained=False)
    # surrogate_model4.fc = torch.nn.Linear(surrogate_model4.fc.in_features, num_classes)

    surrogate_model_path1 = args.save_path_prefix + DataName + '/Pretrain/' + args.surrogate_model1 + '/'
    surrogate_model_path2 = args.save_path_prefix + DataName + '/Pretrain/' + args.surrogate_model2 + '/'
    surrogate_model_path3 = args.save_path_prefix + DataName + '/Pretrain/' + args.surrogate_model3 + '/'
    # surrogate_model_path4 = args.save_path_prefix + DataName + '/Pretrain/' + args.surrogate_model4 + '/'

    model_path1 = os.listdir(surrogate_model_path1)
    for filename in model_path1:
        filepath = os.path.join(surrogate_model_path1, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.pth'):
            print(os.path.join(surrogate_model_path1, filename))
            model_path_resume1 = os.path.join(surrogate_model_path1, filename)
    model_path2 = os.listdir(surrogate_model_path2)
    for filename in model_path2:
        filepath = os.path.join(surrogate_model_path2, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.pth'):
            print(os.path.join(surrogate_model_path2, filename))
            model_path_resume2 = os.path.join(surrogate_model_path2, filename)
    model_path3 = os.listdir(surrogate_model_path3)
    for filename in model_path3:
        filepath = os.path.join(surrogate_model_path3, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.pth'):
            print(os.path.join(surrogate_model_path3, filename))
            model_path_resume3 = os.path.join(surrogate_model_path3, filename)
    # model_path4 = os.listdir(surrogate_model_path4)
    # for filename in model_path4:
    #     filepath = os.path.join(surrogate_model_path4, filename)
    #     if os.path.isfile(filepath) and filename.lower().endswith('.pth'):
    #         print(os.path.join(surrogate_model_path4, filename))
    #         model_path_resume4 = os.path.join(surrogate_model_path4, filename)

    saved_state_dict1 = torch.load(model_path_resume1)
    new_params1 = surrogate_model1.state_dict().copy()
    saved_state_dict2 = torch.load(model_path_resume2)
    new_params2 = surrogate_model2.state_dict().copy()
    saved_state_dict3 = torch.load(model_path_resume3)
    new_params3 = surrogate_model3.state_dict().copy()
    # saved_state_dict4 = torch.load(model_path_resume4)
    # new_params4 = surrogate_model4.state_dict().copy()

    for i, j in zip(saved_state_dict1, new_params1):
        new_params1[j] = saved_state_dict1[i]

    surrogate_model1.load_state_dict(new_params1)

    surrogate_model1 = torch.nn.DataParallel(surrogate_model1).cuda()
    surrogate_model1.eval()

    for i, j in zip(saved_state_dict2, new_params2):
        new_params2[j] = saved_state_dict2[i]

    surrogate_model2.load_state_dict(new_params2)

    surrogate_model2 = torch.nn.DataParallel(surrogate_model2).cuda()
    surrogate_model2.eval()

    for i, j in zip(saved_state_dict3, new_params3):
        new_params3[j] = saved_state_dict3[i]

    surrogate_model3.load_state_dict(new_params3)

    surrogate_model3 = torch.nn.DataParallel(surrogate_model3).cuda()
    surrogate_model3.eval()

    # for i, j in zip(saved_state_dict4, new_params4):
    #     new_params4[j] = saved_state_dict4[i]
    #
    # surrogate_model4.load_state_dict(new_params4)
    #
    # surrogate_model4 = torch.nn.DataParallel(surrogate_model4).cuda()
    # surrogate_model4.eval()

    num_batches = len(imloader)

    kl_loss = torch.nn.KLDivLoss()
    cls_loss = torch.nn.CrossEntropyLoss()
    pgd_loss = torch.nn.KLDivLoss(reduction='sum')
    mse_loss = torch.nn.MSELoss(reduction='none')
    tbar = tqdm(imloader)

    num_iter = 5
    alpha = 1

    if args.attack_func == 'fgsm':
        for batch_index, src_data in enumerate(tbar):
            X, Y, img_name = src_data
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            Y = Y.numpy().squeeze()
            T = torch.tensor([args.target]).cuda()
            tbar.set_description('Batch: %d/%d' % (batch_index + 1, num_batches))
            adv_im.requires_grad = True
            _, out = surrogate_model1(adv_im)
            pred_loss = 0.2 * cls_loss(out, label) - 0.8 * cls_loss(out, T)

            grad = torch.autograd.grad(pred_loss, adv_im,
                                       retain_graph=False, create_graph=False)[0]

            adv_im = adv_im.detach() + args.epsilon * grad / torch.norm(grad, float('inf'))
            delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
            adv_im = (X + delta).detach()

            recreated_image = recreate_image(adv_im.cpu())

            gen_name = img_name[0] + '_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix + gen_name, 'png')

    elif args.attack_func == 'ifgsm':
        for batch_index, src_data in enumerate(tbar):
            X, Y, img_name = src_data
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            Y = Y.numpy().squeeze()
            T = torch.tensor([args.target]).cuda()

            # Start iteration
            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' % (batch_index + 1, num_batches, i + 1))
                adv_im.requires_grad = True
                _, out = surrogate_model1(adv_im)
                pred_loss = 0.2 * cls_loss(out, label) - 0.8 * cls_loss(out, T)
                grad = torch.autograd.grad(pred_loss, adv_im,
                                           retain_graph=False, create_graph=False)[0]

                adv_im = adv_im.detach() + alpha * grad / torch.norm(grad, float('inf'))
                delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
                adv_im = (X + delta).detach()

                recreated_image = recreate_image(adv_im.cpu())
                # Process confirmation image
                adv_im = preprocess_image(Image.fromarray(recreated_image), args)

            gen_name = img_name[0] + '_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix + gen_name, 'png')

    elif args.attack_func == 'pgd':
        for batch_index, src_data in enumerate(tbar):
            X, Y, img_name = src_data
            X = X.cuda()
            adv_im = X.clone().cuda()
            Y = Y.numpy().squeeze()
            T = torch.tensor([args.target]).cuda()
            _, logit_ori = surrogate_model1(X)
            logit_ori = logit_ori.detach()
            # Start iteration
            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' % (batch_index + 1, num_batches, i + 1))
                adv_im.requires_grad = True
                _, logit_adv = surrogate_model1(adv_im)
                pred_loss = 0.2 * pgd_loss(F.log_softmax(logit_adv, dim=1),
                                           F.softmax(logit_ori, dim=1)) - 0.8 * cls_loss(logit_adv, T)

                grad = torch.autograd.grad(pred_loss, adv_im,
                                           retain_graph=False, create_graph=False)[0]

                adv_im = adv_im.detach() + alpha * grad / torch.norm(grad, float('inf'))
                delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
                adv_im = (X + delta).detach()

                recreated_image = recreate_image(adv_im.cpu())
                # Process confirmation image
                adv_im = preprocess_image(Image.fromarray(recreated_image), args)

            gen_name = img_name[0] + '_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix + gen_name, 'png')
    elif args.attack_func == 'cw':
        for batch_index, src_data in enumerate(tbar):
            X, Y, img_name = src_data
            X = X.cuda()
            label_mask = F.one_hot(Y, num_classes=num_classes).cuda()
            T = torch.tensor([args.target]).cuda()
            T1 = F.one_hot(T, num_classes=num_classes).cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            Y = Y.numpy().squeeze()

            # Start iteration
            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' % (batch_index + 1, num_batches, i + 1))
                adv_im.requires_grad = True
                _, out = surrogate_model1(adv_im)

                correct_logit = 0.2 * torch.sum(label_mask * out) - 0.8 * torch.sum(T1 * out)
                wrong_logit = 0.2 * torch.max((1 - label_mask) * out) - 0.8 * torch.max((1 - T1) * out)
                loss = -(correct_logit - wrong_logit + args.C)

                grad = torch.autograd.grad(loss, adv_im,
                                           retain_graph=False, create_graph=False)[0]

                adv_im = adv_im.detach() + alpha * grad / torch.norm(grad, float('inf'))
                delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
                adv_im = (X + delta).detach()

                recreated_image = recreate_image(adv_im.cpu())
                # Process confirmation image
                adv_im = preprocess_image(Image.fromarray(recreated_image), args)

            gen_name = img_name[0] + '_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix + gen_name, 'png')

    elif args.attack_func == 'mixforest' or args.attack_func == 'mixagricultural' or args.attack_func == 'mixairplane' or args.attack_func == 'mixbaseballdiamond' or args.attack_func == 'mixbeach' \
            or args.attack_func == 'mixairport' or args.attack_func == 'mixparking' or args.attack_func == 'mixschool' or args.attack_func == 'mixstadium':
        mix_im = composed_transforms(Image.open(mix_file).convert('RGB')).unsqueeze(0).cuda()
        mix_feature1, _ = surrogate_model1(mix_im)
        mix_feature1 = mix_feature1.data

        mix_feature2, _ = surrogate_model2(mix_im)
        mix_feature2 = mix_feature2.data

        mix_feature3, _ = surrogate_model3(mix_im)
        mix_feature3 = mix_feature3.data
        #
        # mix_feature4, _ = surrogate_model4(mix_im)
        # mix_feature4 = mix_feature4.data
        for batch_index, src_data in enumerate(tbar):
            X, Y, img_name = src_data
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            T = torch.tensor([args.target]).cuda()
            Y = Y.numpy().squeeze()
            momentum = torch.zeros_like(X).cuda()
            # airport 0  forest 11  parking 17  school 24  stadium 27
            # Start iteration
            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' % (batch_index + 1, num_batches, i + 1))
                adv_im.requires_grad = True
                pred_loss1 = 0
                mix_loss1 = 0
                target_loss1 = 0

                pred_loss2 = 0
                mix_loss2 = 0
                target_loss2 = 0

                pred_loss3 = 0
                mix_loss3 = 0
                target_loss3 = 0

                pred_loss4 = 0
                mix_loss4 = 0
                target_loss4 = 0
                for k in range(5):
                    # Scale augmentation
                    feature1, pred1 = surrogate_model1(adv_im / (2 ** (k)))
                    pred_loss1 += cls_loss(pred1, label)
                    mix_loss1 += -kl_loss(feature1, mix_feature1)
                    target_loss1 += -cls_loss(pred1, T)

                    feature2, pred2 = surrogate_model2(adv_im / (2 ** (k)))
                    pred_loss2 += cls_loss(pred2, label)
                    mix_loss2 += -kl_loss(feature2, mix_feature2)
                    target_loss2 += -cls_loss(pred2, T)

                    feature3, pred3 = surrogate_model3(adv_im / (2 ** (k)))
                    pred_loss3 += cls_loss(pred3, label)
                    mix_loss3 += -kl_loss(feature3, mix_feature3)
                    target_loss3 += -cls_loss(pred3, T)

                    # feature4, pred4 = surrogate_model4(adv_im / (2 ** (k)))
                    # pred_loss4 += cls_loss(pred4, label)
                    # mix_loss4 += -kl_loss(feature4, mix_feature4)
                    # target_loss4 += -cls_loss(pred4, T)

                pred_loss = 0.33 * (pred_loss1 + pred_loss2 + pred_loss3)
                mix_loss = 0.33 * (mix_loss1 + mix_loss2 + mix_loss3)
                target_loss = 0.33 * (target_loss1 + target_loss2 + target_loss3)

                total = pred_loss * args.beta + mix_loss + args.gamma * target_loss
                grad = torch.autograd.grad(total, adv_im,
                                           retain_graph=False, create_graph=False)[0]

                grad = grad / torch.norm(grad, p=1)
                grad = grad + momentum * args.decay
                momentum = grad

                adv_im = adv_im.detach() + alpha * grad / torch.norm(grad, float('inf'))
                delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
                adv_im = (X + delta).detach()

                recreated_image = recreate_image(adv_im.cpu())
                adv_im = preprocess_image(Image.fromarray(recreated_image), args)

            gen_name = img_name[0] + '_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix + gen_name, 'png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--surrogate_model', type=str, default='alexnet',
                        help='alexnet,resnet18,densenet121,regnet_x_400mf')
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--root_dir', type=str, default='', help='dataset path.')
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--attack_func', type=str, default='mixforest',
                        help='mixforest，mixagricultural,mixairplane,mixbaseballdiamond,mixbeach,'
                             'mixairport,mixparking,mixschool,mixstadium,fgsm,pgd,cw,ifgsm')
    parser.add_argument('--decay', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--scale', type=float, default=10)
    parser.add_argument('--std', type=float, default=0.5)
    parser.add_argument('--C', type=float, default=50)
    parser.add_argument('--target', type=int, default=27)

    main(parser.parse_args())
