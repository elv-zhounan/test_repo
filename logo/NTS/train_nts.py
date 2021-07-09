from datetime import datetime
import os

import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR

from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, data_root, resume, save_dir
from core import nts, dataset
from core.utils import init_log, progress_bar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_epoch = 1
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)

logging = init_log(save_dir)
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
_print = logging.info
_print('--'*50)
_print('BATCH_SIZE=%s\n\
PROPOSAL_NUM=%s\n\
SAVE_FREQ=%s\n\
LR=%s\n\
WD=%s\n\
data_root=%s\n\
resume=%s\n\
save_dir=%s\n'%(BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, data_root, resume, save_dir))
_print('--'*50)

# read dataset
NUM_WORKERS=2

trainset = dataset.Custom(data_root, is_train=True, data_len=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
testset = dataset.Custom(data_root, is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

_print('Data loading done')

# get num_classes
label_txt_file = open(os.path.join(data_root, 'image_class_labels_500.txt'))
num_classes = 0
for line in label_txt_file:
    num_classes = max(num_classes, int(line[:-1].split(' ')[-1]))
print(num_classes)

# define model
net = nts.AttentionNet(topN=PROPOSAL_NUM, num_classes=num_classes)
if resume:
    ckpt = torch.load(resume, map_location=device)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1

creterion = torch.nn.CrossEntropyLoss()


# for finetuning
#ckpt = torch.load(resume, map_location=device)
#pretrained_dict = ckpt['net_state_dict']
#net_dict = net.state_dict()
#pretrained_dict = {key: val for key, val in pretrained_dict.items() if val.shape == net_dict[key].shape}
#net_dict.update(pretrained_dict)
#net.load_state_dict(net_dict)


_print(net)

# define optimizers
raw_parameters = list(net.pretrained_model.parameters())
part_parameters = list(net.proposal_net.parameters())
concat_parameters = list(net.concat_net.parameters())
partcls_parameters = list(net.partcls_net.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)
schedulers = [MultiStepLR(raw_optimizer, milestones=[15, 30], gamma=0.1),
              MultiStepLR(concat_optimizer, milestones=[15, 30], gamma=0.1),
              MultiStepLR(part_optimizer, milestones=[15, 30], gamma=0.1),
              MultiStepLR(partcls_optimizer, milestones=[15, 30], gamma=0.1)]
net = net.to(device)
net = DataParallel(net)

_print('Start training')
for epoch in range(start_epoch, 200):
    for scheduler in schedulers:
        scheduler.step()

    # begin training
    _print('--' * 50)
    net.train()
    for i, data in enumerate(trainloader):
        img, label = data[0].to(device), data[1].to(device)
        batch_size = img.size(0)
        raw_optimizer.zero_grad()
        part_optimizer.zero_grad()
        concat_optimizer.zero_grad()
        partcls_optimizer.zero_grad()

        raw_logits, concat_logits, part_logits, _, top_n_prob, concat_out = net(img)
        part_loss = nts.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                    label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
        raw_loss = creterion(raw_logits, label)
        concat_loss = creterion(concat_logits, label)
        rank_loss = nts.ranking_loss(top_n_prob, part_loss)
        partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                 label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))
        cosine_loss, _ = nts.cosine_loss(concat_out, label)

        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss + cosine_loss
        total_loss.backward()
        raw_optimizer.step()
        part_optimizer.step()
        concat_optimizer.step()
        partcls_optimizer.step()
        progress_bar(i, len(trainloader), 'train')

    if epoch % SAVE_FREQ == 0:
        net.eval()
        #train_loss = 0
        #train_correct = 0
        #total = 0
        #pair = 0
        #cosine_loss = 0
        #for i, data in enumerate(trainloader):
        #    with torch.no_grad():
        #        img, label = data[0].to(device), data[1].to(device)
        #        batch_size = img.size(0)
        #        _, concat_logits, _, _, _, concat_out = net(img)
        #        # calculate loss
        #        concat_loss = creterion(concat_logits, label)
        #        # calculate accuracy
        #        _, concat_predict = torch.max(concat_logits, 1)
        #        total += batch_size
        #        train_correct += torch.sum(concat_predict.data == label.data)
        #        train_loss += concat_loss.item() * batch_size
        #        curr_cosine_loss, mask_sum = nts.cosine_loss(concat_out, label)
        #        cosine_loss += curr_cosine_loss.item() * mask_sum.item()
        #        pair += mask_sum.item()
        #        progress_bar(i, len(trainloader), 'eval train set')

        #train_acc = float(train_correct) / total
        #train_loss = train_loss / total
        #cosine_loss = cosine_loss / pair

        #_print(
        #    'epoch:{} - train loss: {:.3f}, cosine_loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
        #        epoch,
        #        train_loss,
        #        cosine_loss,
        #        train_acc,
        #        total))

	# evaluate on test set
        test_loss = 0
        test_correct = 0
        total = 0
        pair = 0
        cosine_loss = 0
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, label = data[0].to(device), data[1].to(device)
                batch_size = img.size(0)
                _, concat_logits, _, _, _, concat_out = net(img)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                test_correct += torch.sum(concat_predict.data == label.data)
                test_loss += concat_loss.item() * batch_size
                curr_cosine_loss, mask_sum = nts.cosine_loss(concat_out, label)
                cosine_loss += curr_cosine_loss.item() * mask_sum.item()
                pair += mask_sum.item()
                progress_bar(i, len(testloader), 'eval test set')

        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        cosine_loss = cosine_loss / pair
        _print(
            'epoch:{} - test loss: {:.3f}, cosine loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                epoch,
                test_loss,
                cosine_loss,
                test_acc,
                total))

	# save model
        net_state_dict = net.module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            #'train_loss': train_loss,
            #'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'cosine_loss': cosine_loss,
            'net_state_dict': net_state_dict},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

_print('Finishing training')
