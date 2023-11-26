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
            mix_file = 'H:/code/UAE-RS-main/dataset1/UCM_' + args.attack_func + '_sample.png'
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
    if args.surrogate_model == 'alexnet':
        surrogate_model = models.alexnet(pretrained=False)
        surrogate_model.classifier._modules['6'] = nn.Linear(4096, num_classes)
    elif args.surrogate_model == 'resnet18':
        surrogate_model = models.resnet18(pretrained=False)
        surrogate_model.fc = torch.nn.Linear(surrogate_model.fc.in_features, num_classes)
    elif args.surrogate_model == 'densenet121':
        surrogate_model = models.densenet121(pretrained=False)
        surrogate_model.classifier = nn.Linear(1024, num_classes)
    elif args.surrogate_model == 'regnet_x_400mf':
        surrogate_model = models.regnet_x_400mf(pretrained=False)
        surrogate_model.fc = torch.nn.Linear(surrogate_model.fc.in_features, num_classes)
    elif args.surrogate_model == 'regnet_x_16gf':
        surrogate_model = models.regnet_x_16gf(pretrained=False)
        surrogate_model.fc = torch.nn.Linear(surrogate_model.fc.in_features, num_classes)

    surrogate_model_path = args.save_path_prefix + DataName + '/Pretrain/' + args.surrogate_model + '/'
    model_path = os.listdir(surrogate_model_path)
    for filename in model_path:
        filepath = os.path.join(surrogate_model_path, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.pth'):
            print(os.path.join(surrogate_model_path, filename))
            model_path_resume = os.path.join(surrogate_model_path, filename)

    saved_state_dict = torch.load(model_path_resume)
    new_params = surrogate_model.state_dict().copy()

    for i, j in zip(saved_state_dict, new_params):
        new_params[j] = saved_state_dict[i]

    surrogate_model.load_state_dict(new_params)

    surrogate_model = torch.nn.DataParallel(surrogate_model).cuda()
    surrogate_model.eval()

    num_batches = len(imloader)

    kl_loss = torch.nn.KLDivLoss()
    cls_loss = torch.nn.CrossEntropyLoss()
    tbar = tqdm(imloader)
    pgd_loss = torch.nn.KLDivLoss(reduction='sum')

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
            _, out = surrogate_model(adv_im)
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
                _, out = surrogate_model(adv_im)
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
            _, logit_ori = surrogate_model(X)
            logit_ori = logit_ori.detach()
            # Start iteration
            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' % (batch_index + 1, num_batches, i + 1))
                adv_im.requires_grad = True
                _, logit_adv = surrogate_model(adv_im)
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
                _, out = surrogate_model(adv_im)

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
        mix_feature, _ = surrogate_model(mix_im)
        mix_feature = mix_feature.data
        for batch_index, src_data in enumerate(tbar):
            X, Y, img_name = src_data
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            T = torch.tensor([args.target]).cuda()
            Y = Y.numpy().squeeze()
            momentum = torch.zeros_like(X).cuda()

            # Start iteration
            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' % (batch_index + 1, num_batches, i + 1))
                adv_im.requires_grad = True
                pred_loss = 0
                mix_loss = 0
                target_loss = 0
                for k in range(5):
                    # Scale augmentation
                    feature, pred = surrogate_model(adv_im / (2 ** (k)))
                    pred_loss += cls_loss(pred, label)
                    mix_loss += -kl_loss(feature, mix_feature)
                    target_loss += -cls_loss(pred, T)
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
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--scale', type=float, default=10)
    parser.add_argument('--std', type=float, default=0.5)
    parser.add_argument('--C', type=float, default=50)
    parser.add_argument('--target', type=int, default=27)
    # AID airport 0  forest 11  parking 17  school 24  stadium 27
    # UCM agricultural 0  airplane 1  baseballdiamond 2  beach 3  forest 7

    main(parser.parse_args())
