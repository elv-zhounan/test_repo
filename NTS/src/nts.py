from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from core import resnet
import numpy as np
from core.anchors import generate_default_anchor_maps, hard_nms
from config import CAT_NUM, PROPOSAL_NUM
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)


class AttentionNet(nn.Module):
    def __init__(self, topN=PROPOSAL_NUM, num_classes=2000):
        super(AttentionNet, self).__init__()
        self.pretrained_model = resnet.resnet152(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, num_classes)
        self.proposal_net = ProposalNet()
        self.topN = topN
        self.concat_net = nn.Linear(2048 * (CAT_NUM + 1), num_classes)
        self.partcls_net = nn.Linear(512 * 4, num_classes)
        _, edge_anchors, areas = generate_default_anchor_maps()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)

    def forward(self, x):
        # default shape of x: (-1, 448, 448, 3)
        # Resnet extract feature, later on the 'navigator(aka proposal_net)' will use it
        resnet_out, rpn_feature, feature = self.pretrained_model(x)
        # resnet_out: (-1, 2000) rpn_feature: (-1, 14, 14, 2048), feature (-1, 2048)  PS.of course of torch, channel will at the front
        
        # x shape in torch (-1, 3, 448, 448), after padding (-1, 3, 996, 996)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)

        # feed rpn_feature(-1, 14, 14, 2048) into proposal_net, output shape : get three feature maps as output:
        # [-1, 6, 14, 14] [-1, 6, 7, 7] [-1, 9, 4, 4]
        # each coresponds to an anchor box's infomation, according to the paper
        # then we reshape each to (batch_size, -1) and concat them to be a (batch_size, 1614) shape tensor
        rpn_score = self.proposal_net(rpn_feature.detach())

        # for each sample in this batch, we get its rpn_score, reshape to (1614,1) then concat it with edge_anchors(1614, 4), and another tensor used to show the region's index, which shape is (1614, 1)
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        # finally the all_cdds is a list with a length of batch_size and each element in the list has a shape of (1614, 6)

        # for each sample , we do NMS, and get topN region for each sample
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]

        # make it to a numpy array again
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).to(device) # make top_n_index to a tensor for future training use

        # according to the index, we get the rpn_score of those topn_cdds
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)

        # get the sub_img in the cdd_boxes, going to feed them into teacher network
        part_imgs = torch.zeros([batch, self.topN, 3, 224, 224]).to(device)
        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)

        # feed the part_imgs into the pretained_model (aka renet) to extract features
        part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)
        _, _, part_features = self.pretrained_model(part_imgs.detach())
        # part_features shape: (batch * self.topN, 2048)

        # and reshape it
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        # part_feature shape: (batch_size, CAT_NUM*2048)

        # concat the features that teacher-net learnt with the features of the input_img from the very beginning
        concat_out = torch.cat([part_feature, feature], dim=1)
        # concat_out shape: (batch_size, (CAT_NUM+1)*2048)

        # a linear layer prediction based on the concat_feature
        concat_logits = self.concat_net(concat_out)
        
        # the prediction from the ResNet
        raw_logits = resnet_out

        # part_logits have the shape: B*N*NUM_CLASSES
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        # part_logits shape: (batch, self.topN, 2000)

        # raw_logits: the resnet output with the input_img as input
        # concat_logits: prediction based on the concat feature
        # part_logits: the prediction for each of the TOP_N_cdds 
        # concat_out: the concat feature

        
        return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob, concat_out]


def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)


def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
    loss = Variable(torch.zeros(1).to(device))
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size

def cosine_loss(features, labels, eps = 1e-8):
    top = torch.mm(features, torch.transpose(features, 0, 1))
    features_norm = torch.norm(features, dim=1)
    bot = torch.clamp(torch.mm(features_norm.unsqueeze(1), features_norm.unsqueeze(0)), min=eps)
    cos_sim = top / bot

    batch_size = features.size(0)
    mask = torch.zeros([batch_size, batch_size]).to(device)
    for i in range(batch_size):
        for j in range(batch_size):
            mask[i,j] = (labels[i] != labels[j])

    loss = Variable(torch.zeros(1).to(device))
    loss += torch.sum(torch.exp(cos_sim) * mask)
    loss /= torch.clamp(mask.sum(), min=eps)

    return loss, torch.clamp(mask.sum(), min=eps)
