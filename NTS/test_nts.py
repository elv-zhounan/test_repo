import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

from torch.autograd import Variable
import torch.utils.data
import torch
from torch.nn import DataParallel
from config import BATCH_SIZE, PROPOSAL_NUM, test_model, data_root
from core import nts, dataset
from core.utils import progress_bar


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not test_model:
    raise NameError('please set the test_model file to choose the checkpoint!')

# read dataset
NUM_WORKERS=1

testset = dataset.Custom(data_root, is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

print('Data loading done!')

# get num_classes
label_txt_file = open(os.path.join(data_root, 'image_class_labels.txt'))
num_classes = 0
for line in label_txt_file:
    num_classes = max(num_classes, int(line[:-1].split(' ')[-1]))
print(num_classes)

# define model
net = nts.AttentionNet(topN=PROPOSAL_NUM, num_classes=num_classes)
ckpt = torch.load(test_model, map_location=device)
net.load_state_dict(ckpt['net_state_dict'])
net = net.to(device)
# net = DataParallel(net)
creterion = torch.nn.CrossEntropyLoss()

print('Evaluating model {}'.format(test_model))


if __name__ == '__main__':

    torch.multiprocessing.freeze_support()
    # evaluate on test set
    net.eval()
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
            # print(concat_logits.size(), label.size(), concat_loss.size())
            # calculate accuracy
            _, concat_predict = torch.max(concat_logits, 1)
            total += batch_size
            test_correct += torch.sum(concat_predict.data == label.data)
            test_loss += concat_loss.item() * batch_size
            curr_cosine_loss, mask_sum = nts.cosine_loss(concat_out, label)
            cosine_loss += curr_cosine_loss.item() * mask_sum.item()
            pair += mask_sum.item()
            progress_bar(i, len(testloader), 'eval on test set')
            if i == 3:
                break

    test_acc = float(test_correct) / total
    test_loss = test_loss / total
    cosine_loss = cosine_loss / pair
    print('Test set loss: {:.3f}, cosine loss: {:.3f} and test set acc: {:.3f} total sample: {}'.format(test_loss, cosine_loss, test_acc, total))

    print('Finishing testing')
