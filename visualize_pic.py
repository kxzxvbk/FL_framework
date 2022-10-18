import copy

from models.moco import MoCo
from utils.dataset import DatasetConstructor
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from utils.dataset import GaussianBlur
import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight


    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])  # 算出总共求了多少次差
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class Simulator:
    """
    a simulator for a single device to perform federated learning
    """

    def get_raw_image(self, idx):
        example = self.dataset[idx][0]
        return copy.deepcopy(example)

    def get_aug_image(self, idx):
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 0.21)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=1.),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.),
            transforms.RandomHorizontalFlip(),
        ]
        transform = transforms.Compose(augmentation)

        example = transform(self.dataset[idx][0])
        return copy.deepcopy(example)

    def get_rec_image(self, aug_img):
        augmentation = [
            transforms.ToTensor()
        ]
        transform = transforms.Compose(augmentation)
        in_img = transform(aug_img).unsqueeze(0)

        for k, param in enumerate(self.model.parameters()):
            param.requires_grad = False
        self.model.eval()
        # self.model.cuda()
        # out_feature = nn.functional.normalize(self.model.encoder_q(in_img), dim=-1).detach()
        out_feature = self.model.encoder_q(in_img).detach()

        rec_image = torch.randn(in_img.shape)
        rec_image = torch.nn.Parameter(rec_image, requires_grad=True)
        opt = torch.optim.Adam([rec_image], lr=1e-2, weight_decay=1e-4)
        criter = torch.nn.L1Loss()
        tv_loss = TVLoss()
        for iteration in range(6000):
            if iteration != 0 and iteration % 2000 == 0:
                opt.param_groups[0]['lr'] /= 10
            rec_feature = self.model.encoder_q(rec_image)
            # rec_feature = nn.functional.normalize(rec_feature, dim=-1)
            tv = tv_loss(rec_image)
            mse = criter(rec_feature, out_feature)
            loss = mse + 0.2 * tv
            if iteration % 100 == 0:
                print(mse.item(), tv.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        return (255 * rec_image[0].detach().numpy().transpose(1, 2, 0)).astype('uint8')

    def __init__(self, args):
        self.args = args
        self.dataset = None

    def plot_imgs(self, raw_img, aug_img, rec_img):
        # plot 1:

        plt.subplot(1, 3, 1)
        plt.imshow(raw_img)
        plt.title("Raw Image")

        # plot 2:
        plt.subplot(1, 3, 2)
        plt.imshow(aug_img)
        plt.title("Augmented Image")

        # plot 3:
        plt.subplot(1, 3, 3)
        plt.imshow(rec_img)
        plt.title("Recovered Image")

        plt.show()

    def _init_network(self):
        self.model = MoCo(client_id=-1, use_global_queue=False)
        sd = torch.load(path)
        self.model.load_state_dict(sd, strict=False)

    def run(self):
        # load dataset
        self.dataset = DatasetConstructor(self.args).get_dataset()
        self._init_network()

        raw_img = self.get_raw_image(7344)
        aug_img = self.get_aug_image(7344)
        rec_img = self.get_rec_image(aug_img)
        self.plot_imgs(raw_img, aug_img, rec_img)


if __name__ == '__main__':
    from args.moco_vis import args_parser
    path = './model_checkpoints/moco_model.ckpt'
    args = args_parser()
    simulator = Simulator(args)
    simulator.run()
