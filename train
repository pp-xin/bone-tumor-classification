import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from networks import resnet_bn, resnet_gn,Unet3D,resnet_2D,resnet_2D_3
from networks.Non_Local import resnet3D
import csv
import pandas as pd
from misc import printProgressBar
from torch.optim import lr_scheduler
from  util import GradualWarmupScheduler
from gcam import gcam
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        F_loss_all = 0
        for i in range(len(inputs)):
            BCE_loss = F.binary_cross_entropy(inputs[i], targets[i], reduce=False)
        # if self.logits:
        #     BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        # else:
        #     BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
            pt = torch.exp(-BCE_loss)
            if targets[i] == 1:
                F_loss = 0.75 *5*(1-pt)**self.gamma * BCE_loss
            if targets[i] == 0:
                F_loss = 0.25 *2*(1-pt)**self.gamma * BCE_loss
            F_loss_all += F_loss
        return  torch.mean(F_loss_all)
        #
        # if self.reduce:
        #     return torch.mean(F_loss)
        # else:
        #     return F_loss


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader, extra_loader=None,extra_loader2=None):
        # Make record file
        if config.mode == 'train':
            self.record_file = os.path.join(config.result_path, 'record.txt')
        else:
            self.record_file = os.path.join(config.result_path, 'record_t.txt')
        f = open(self.record_file, 'w')
        f.close()

        self.Task_name = config.Task_name

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.extra_loader = extra_loader
        self.extra_loader2 = extra_loader2

        self.train_list = config.train_list
        self.valid_list = config.valid_list
        self.test_list = config.test_list
        self.extra_list = config.extra_list
        self.extra_list2 = config.extra_list2
        self.with_extra = config.with_extra_data

        # Models
        self.classnet = None
        self.optimizer = None
        self.img_size = config.image_size
        self.patch_size = config.patch_size
        # self.img_ch = config.img_ch'
        # self.output_ch = config.output_ch
        self.criterion = torch.nn.BCELoss()
        # self.criterion = FocalLoss()
        #

        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # learning rate
        self.num_epochs_decay = config.num_epochs_decay
        self.decay_ratio = config.decay_ratio
        self.decay_step = config.decay_step
        self.lr_low = config.lr_low
        self.lr_cos_epoch = config.lr_cos_epoch
        self.lr_warm_epoch = config.lr_warm_epoch
        self.lr_sch = None  # 初始化先设置为None
        self.lr_list = []  # 临时记录lr
        self.loss_list = []
        self.best_epoch = 0
        self.best_classnet_score = 0

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        self.save_detail_result = config.save_detail_result
        self.log_dir = config.log_dir

        #result
        self.fpr_all = []
        self.tpr_all = []
        self.auc_all = []
        self.fold_num = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cuda:'+str(1-config.cuda_idx) if torch.cuda.is_available() else 'cpu') #dont know why 1-idx
        self.DataParallel = config.DataParallel
        self.model_type = config.model_type

        self.pre_threshold = 0.5

        self.my_init()

    def myprint(self, *args):
        """Print & Record while training."""
        print(*args)
        f = open(self.record_file, 'a')
        print(*args, file=f)
        f.close()

    def my_init(self):
        self.myprint(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))
        self.print_date_msg()
        self.build_model()

    def print_date_msg(self):
        self.myprint("patient count in train:{}".format(len(self.train_list)), self.train_list)
        self.myprint("patient count in valid:{}".format(len(self.valid_list)), self.valid_list)
        self.myprint("patient count in test :{}".format(len(self.test_list)), self.test_list)
        self.myprint("patient count in extra :{}".format(len(self.extra_list)), self.extra_list)
        self.myprint("patient count in extra2 :{}".format(len(self.extra_list2)), self.extra_list2)


    def build_model(self):  # todo
        """Build generator and discriminator."""
        if self.model_type == 'resnet_bn':
            self.classnet = resnet_bn.resnet34(
                num_classes=1,
                sample_size=self.img_size,
                sample_duration=self.patch_size)
        elif self.model_type == 'efficientnet':
            self.classnet = EfficientNet.from_name('efficientnet-b6')
            state_dict = torch.load('/data/LHX/efficientnet/efficientnet-b6-c76e70fd.pth')
            self.classnet.load_state_dict(state_dict)
            features = self.classnet._fc.in_features
            self.classnet._fc = nn.Linear(in_features=features, out_features=1,bias=True)
        elif self.model_type == 'unet3d':
            self.classnet = Unet3D.UNet(1,1)
        elif self.model_type == 'resnet_2D':
            self.classnet = resnet_2D.resnet50(num_classes=1,sample_size=0,sample_duration=0)
        elif self.model_type == 'resnet_2D_3':
            self.classnet = resnet_2D_3.resnet50(num_classes=1, sample_size=0, sample_duration=0)
        elif self.model_type == 'resnet_bn_152':
            self.classnet = resnet_bn.resnet152(
                num_classes=1,
                sample_size=self.img_size,

                sample_duration=self.patch_size)
        elif self.model_type == 'resnet_gn':
            self.classnet = resnet_gn.resnet50(
                num_classes=1,
                sample_size=self.img_size,
                sample_duration=self.patch_size)


        self.classnet.to(self.device)
        if self.DataParallel:
            self.classnet = torch.nn.DataParallel(self.classnet)
        self.print_network(self.classnet, self.model_type)

        # 优化器修改
        self.optimizer = optim.Adam(list(self.classnet.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        #self.optimizer = optim.SGD(list(self.classnet.parameters()),self.lr,0.9)



        # lr schachle策略(要传入optimizer才可以)
        # 暂时的三种情况,(1)只用cos,(2)只用warmup,(3)两者都用
        if self.lr_warm_epoch != 0 and self.lr_cos_epoch == 0:  # zhishiyong
            self.update_lr(self.lr_low)  # 使用warmup需要吧lr初始化为最小lr
            self.lr_sch = GradualWarmupScheduler(self.optimizer,
                                                 multiplier=self.lr / self.lr_low,
                                                 total_epoch=self.lr_warm_epoch,
                                                 after_scheduler=None)
            print('use warmup lr sch')
        elif self.lr_warm_epoch == 0 and self.lr_cos_epoch != 0:
            self.lr_sch = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                         self.lr_cos_epoch,
                                                         eta_min=self.lr_low)
            print('use cos lr sch')
        elif self.lr_warm_epoch != 0 and self.lr_cos_epoch != 0:
            self.update_lr(self.lr_low)  # 使用warmup需要吧lr初始化为最小lr
            # scheduler_cos = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
            #                                                          T_0 = self.lr_warm_epoch,
            #                                                          T_mult= 2 ,
            #                                                          eta_min=self.lr_low)
            scheduler_cos = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                           self.lr_cos_epoch,
                                                           eta_min=self.lr_low)
            self.lr_sch = GradualWarmupScheduler(self.optimizer,
                                                 multiplier=self.lr / self.lr_low,
                                                 total_epoch=self.lr_warm_epoch,
                                                 after_scheduler=scheduler_cos)
            print('use warmup and cos lr sch')
        else:
            if self.lr_sch is None:
                print('use decay coded by dasheng')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        self.myprint(model)
        self.myprint(name)
        self.myprint("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, lr):
        """Update the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.classnet.zero_grad()


    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        """Train encoder, generator and discriminator."""
        print('-----------------------%s-----------------------------' % self.Task_name)
        # ====================================== Training ===========================================#
        # ===========================================================================================#

        # classnet_path = '/home/szu/liver/final_nii_data/model.pkl'
        # self.classnet.load_state_dict(torch.load(classnet_path))
        # self.optimizer.param_groups[0]['lr'] = self.lr
        # print('%s is Successfully Loaded from %s'%(self.model_type,classnet_path))
        # # classnet_path = os.path.join(self.model_path, 'epoch40_Testdice0.7425.pkl')
        # #classnet_path = ('/home/szu/liver/result/CAM_test_1_5/models/epoch50_Test auc0.7726.pkl')
        writer = SummaryWriter(log_dir=self.log_dir)
        #
        # # classnet Train
        # if os.path.isfile(classnet_path):
        # 	# Load the pretrained Encoder
        # 	self.classnet.load_state_dict(torch.load(classnet_path))
        # 	# self.optimizer.param_groups[0]['lr'] = self.lr
        # 	print('%s is Successfully Loaded from %s'%(self.model_type,classnet_path))

        # Train for Encoder


        Iter = 0
        epoch_init = 0
        b = 0.15
        train_len = len(self.train_loader)
        valid_record = np.zeros((1, 7))  # [epoch, Iter, acc, SE, SP, threshold, AUC]
        test_record = np.zeros((1, 7))  # [epoch, Iter, acc, SE, SP, threshold, AUC]
        extra_record = np.zeros((1, 7))  # [epoch, Iter, acc, SE, SP, threshold, AUC]

        self.myprint('Training...')
        for epoch in range(epoch_init, self.num_epochs):
            self.classnet.train(True)
            epoch_loss = 0
            length = 0

            # for i, (_, patch_data, label) in enumerate(self.train_loader):
            for i, sample in enumerate(self.train_loader):
                (_, patch_data, label) = sample
                patch_data = patch_data.to(self.device)  # (N, C_{in}, D_{in}, H_{in}, W_{in})
                # print(patch_data.shape)
                label = label.to(self.device)
                # print(label)

                # Pre : Prediction Result
                pre_probs = self.classnet(patch_data)
                pre_probs = F.sigmoid(pre_probs)
                # pre_probs = F.sigmoid(pre_probs)#todo
                pre_flat = pre_probs.view(-1)
                # print(pre_flat)
                label_flat = label.view(-1)
                # print(label_flat)
                loss = self.criterion(pre_flat, label_flat)
                # print(loss)
                epoch_loss += loss.item()
                # epoch_loss += float(loss)

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                length += 1
                Iter += 1
                writer.add_scalars('Loss', {'train': loss}, Iter)

                # trainning bar
                current_lr = self.optimizer.param_groups[0]['lr']
                print_content = 'learning_rate:' + str(current_lr) + ' batch_loss:' + str(loss.data.cpu().numpy())
                printProgressBar(i + 1, train_len, content=print_content)

            epoch_loss = epoch_loss / length
            self.myprint('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, self.num_epochs, epoch_loss))
            writer.add_scalars('Learning rate', {'lr': current_lr}, epoch)
            self.lr_list.append(current_lr)
            self.loss_list.append(epoch_loss)
            # 保存lr为png
            figg = plt.figure(1)
            plt.plot(self.lr_list)
            figg.savefig(os.path.join(self.result_path, 'lr.PNG'))
            plt.close()
            figg2 = plt.figure(2)
            plt.plot(self.loss_list)
            figg2.savefig(os.path.join(self.result_path, 'loss.PNG'))
            plt.close()

            # Test or_train
            # acc, SE, SP, AUC= self.test(mode='train') # TODO 这里的测试依然是使用了扩增之后的训练集，理应不进行扩增
            # writer.add_scalars('Train', {'Dice': DC}, epoch)
            # self.myprint('[Training]   Acc: %.4f, SE: %.4f, SP: %.4f, AUC: %.4f' % (acc, SE, SP, AUC))
            # self.lr_sch.step()
            # learning rate
            # lr scha way 1:
            if self.lr_sch is not None:
                if (epoch + 1) <= (self.lr_cos_epoch + self.lr_warm_epoch):
                    self.lr_sch.step()
                else :
                    self.lr_sch = None
            # lr scha way 2: Decay learning rate(如果使用方式1,则不使用此方式)
            if self.lr_sch is None:
                if ((epoch + 1) >= self.num_epochs_decay) and (
                        (epoch + 1 - self.num_epochs_decay) % self.decay_step == 0):
                    if current_lr >= self.lr_low:
                        self.lr = current_lr * self.decay_ratio
                        # self.lr /= 100.0
                        self.update_lr(self.lr)
                        self.myprint('Decay learning rate to lr: {}.'.format(self.lr))

            if (epoch + 1) % self.val_step == 0  :
                # ===================================== Validation ====================================#
                acc, SE, SP, threshold, AUC, cost = self.test(mode='valid', save_detail_result=self.save_detail_result,
                                                              during_Trianing=True,current_epoch=epoch+1)
                valid_record = np.vstack((valid_record, np.array([epoch + 1, Iter, acc, SE, SP, threshold, AUC])))
                classnet_score = AUC
                writer.add_scalars('Loss', {'test': cost}, Iter)
                writer.add_scalars('Valid', {'AUC': AUC}, epoch)
                self.myprint('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (
                acc, SE, SP, threshold, AUC))
                save_classnet = self.classnet.state_dict()

                # if classnet_score > self.best_classnet_score or AUC > 0.7:
                #     torch.save(save_classnet,
                #                os.path.join(self.model_path, 'epoch%d_Test auc%.4f.pkl' % (epoch + 1, AUC)))
                #     self.best_classnet_score = classnet_score
                #     self.best_epoch = epoch +1
                #     best_classnet = self.classnet.state_dict()
                #     self.myprint(
                #         'Best %s model in epoch %d, score : %.4f' % (self.model_type, self.best_epoch, self.best_classnet_score))

                if AUC > 0.7 and self.with_extra:

                    acc, SE, SP, threshold, AUC, cost = self.test(mode='extra',
                                                                  save_detail_result=self.save_detail_result,
                                                                  during_Trianing=True,current_epoch=epoch+1)
                    if AUC > 0.7:
                        torch.save(save_classnet,os.path.join(self.model_path, 'epoch%d_Test auc%.4f.pkl' % (epoch + 1, AUC)))
                    extra_record = np.vstack((extra_record, np.array([epoch + 1, Iter, acc, SE, SP, threshold, AUC])))
                    writer.add_scalars('Loss', {'extra': cost}, Iter)
                    writer.add_scalars('Extra', {'AUC': AUC}, epoch)
                    self.myprint('[Extra]      Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, EXT_AUC: %.4f' % (acc, SE, SP, threshold, AUC))

                    acc, SE, SP, threshold, AUC, cost = self.test(mode='extra2',
                                                                  save_detail_result=self.save_detail_result,
                                                                  during_Trianing=True,current_epoch=epoch+1)
                    # if AUC > 0.7:
                    #     torch.save(save_classnet,os.path.join(self.model_path, 'epoch%d_Test auc%.4f.pkl' % (epoch + 1, AUC)))
                    extra_record = np.vstack((extra_record, np.array([epoch + 1, Iter, acc, SE, SP, threshold, AUC])))
                    writer.add_scalars('Loss', {'extra2': cost}, Iter)
                    writer.add_scalars('Extra2', {'AUC': AUC}, epoch)
                    self.myprint('[Extra2]      Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, EXT_AUC: %.4f' % (acc, SE, SP, threshold, AUC))

                # Save Best class model

                    # save_classnet = self.classnet.state_dict()
                    # torch.save(save_classnet,
                    #            os.path.join(self.model_path, 'epoch%d_Test auc%.4f.pkl' % (epoch + 1, AUC)))
                    # acc, SE, SP, threshold, AUC, cost = self.test(mode='extra',
                    #                                               save_detail_result=self.save_detail_result,
                    #                                               during_Trianing=True)
                    # extra_record = np.vstack((extra_record, np.array([epoch + 1, Iter, acc, SE, SP, threshold, AUC])))
                    # writer.add_scalars('Loss', {'extra': cost}, Iter)
                    # writer.add_scalars('Extra', {'AUC': AUC}, epoch)
                    # self.myprint('[Extra]      Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (
                    # acc, SE, SP, threshold, AUC))
                    # torch.save(best_classnet,
                    #            os.path.join(self.model_path, 'epoch%d_Test auc%.4f.pkl' % (epoch + 1, AUC)))

                # ===================================== Test ====================================#


                # save_record_in_xlsx
                if (True):
                    excel_save_path = os.path.join(self.result_path, 'record.xlsx')
                    record = pd.ExcelWriter(excel_save_path)
                    detail_result1 = pd.DataFrame(valid_record)
                    detail_result1.to_excel(record, 'valid', float_format='%.5f')
                    # detail_result2 = pd.DataFrame(test_record)
                    # detail_result2.to_excel(record, 'test', float_format='%.5f')
                    if self.with_extra:
                        detail_result3 = pd.DataFrame(extra_record)
                        detail_result3.to_excel(record, 'extra', float_format='%.5f')
                    record.save()
                    record.close()
        # path = os.path.join(self.model_path,'epoch'+self.best_epoch+'_Test auc'+self.best_classnet_score+'.pkl')
        # acc, SE, SP, threshold, AUC,fpr,tpr = self.test_or(mode='valid',classnet_path=path,save_detail_result=self.save_detail_result)
        self.myprint('Finished!')
        self.myprint(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))

    def test(self, mode='train', classnet_path=None, save_detail_result=False, during_Trianing=False,current_epoch = 0):
        """Test model & Calculate performances."""
        if not classnet_path is None:
            # if os.path.isfile(classnet_path):
            self.classnet.load_state_dict(torch.load(classnet_path))
            self.myprint('%s is Successfully Loaded from %s' % (self.model_type, classnet_path))

        self.classnet.train(False)
        self.classnet.eval()

        if mode == 'train':
            data_lodear = self.train_loader
        elif mode == 'test':
            data_lodear = self.test_loader
        elif mode == 'valid':
            data_lodear = self.valid_loader
        elif mode == 'extra':
            if self.extra_loader is None:
                print('Extra data is not existed!!')
                return
            data_lodear = self.extra_loader
        elif mode == 'extra2':
            if self.extra_loader is None:
                print('Extra data is not existed!!')
                return
            data_lodear = self.extra_loader2
        # else:
        # 	ValueError

        # model pre for each patch
        patient_order_list = []
        patch_order_list = []
        pre_list = []
        label_list = []
        cost = 0.0
        for i, sample in enumerate(data_lodear):
            (patch_paths, patch, label) = sample
            patch_paths = list(patch_paths)
            with torch.no_grad():
                patch = patch.to(self.device)
                label = label.to(self.device)
                pre_probs = self.classnet(patch)
                pre_probs = F.sigmoid(pre_probs)#todo

                pre_flat = pre_probs.view(-1)
                # print(pre_flat)
                label_flat = label.view(-1)
                # print(patch_paths,label_flat)
                loss = self.criterion(pre_flat, label_flat)
                cost += float(loss)

            pre_probs = pre_probs.data.cpu().numpy()
            label = label.data.cpu().numpy()

            for ii in range(pre_probs.shape[0]):
                pre_tmp = pre_probs[ii, :]
                # label_tmp = label[ii, :]
                label_tmp = label[ii]

                pre_list.append(pre_tmp.reshape(-1))
                label_list.append(label_tmp.reshape(-1))

                tmp_index = patch_paths[ii].split('/')[-1]
                tmp_index1 = tmp_index.split('_')[0][:]
                patient_order_list.append(int(tmp_index1))
                tmp_index2 = tmp_index.split('_')[1][:]
                tmp_index2 = tmp_index2.split('.')[0][:]
                patch_order_list.append(int(tmp_index2))

        cost /= (i + 1)

        detail_result1 = np.zeros([len(patient_order_list), 4])  # detail_msg = [id, patch_id, pre, label]
        detail_result1[:, 0] = np.array(patient_order_list).T
        detail_result1[:, 1] = np.array(patch_order_list).T
        detail_result1[:, 2] = np.array(pre_list).T
        detail_result1[:, 3] = np.array(label_list).T

        # statistic for each patient
        patinet_order_unique = np.unique(patient_order_list)
        detail_result2 = np.zeros([len(patinet_order_unique), 4])  # detail_msg = [id, _, mpre, label]
        detail_result2[:, 0] = patinet_order_unique.T
        for unique_p_order in patinet_order_unique:
            select_patient_index = [i for i, x in enumerate(patient_order_list) if x == unique_p_order]
            pre_tmp = []
            label_tmp = []
            for i in select_patient_index:
                pre_tmp.append(pre_list[i])
                label_tmp.append(label_list[i])
            pre_probs = np.array(pre_tmp).reshape(-1)
            label = np.array(label_tmp).reshape(-1)
            if mode == 'valid':
                mean_pre_probs = np.max(pre_probs)  # TODO
            if mode == 'extra':
                mean_pre_probs = np.max(pre_probs)
            if mode == 'extra2':
                mean_pre_probs = np.max(pre_probs)
            label = np.mean(label)

            # detail_result[detail_result[:,0] == unique_p_order,1] = get_AUC(pre_probs, label)
            detail_result2[detail_result2[:, 0] == unique_p_order, 2] = mean_pre_probs
            detail_result2[detail_result2[:, 0] == unique_p_order, 3] = label

        P_pre_probs = torch.from_numpy(detail_result2[:, 2]).to(self.device)
        P_label = torch.from_numpy(detail_result2[:, 3]).to(self.device)

        if mode == 'train':
            threshold = get_bset_threshold(P_pre_probs, P_label)
        elif mode == 'valid':
            threshold = get_bset_threshold(P_pre_probs, P_label)
            self.pre_threshold = threshold
        elif mode == 'extra':
            threshold = get_bset_threshold(P_pre_probs, P_label)
            self.pre_threshold = threshold
        elif mode == 'extra2':
            threshold = get_bset_threshold(P_pre_probs, P_label)
            self.pre_threshold = threshold
            # threshold = self.pre_threshold
        else:
            threshold = self.pre_threshold
        accuracy = get_accuracy(P_pre_probs, P_label, threshold)
        sensitivity = get_sensitivity(P_pre_probs, P_label, threshold)
        specificity = get_specificity(P_pre_probs, P_label, threshold)
        AUC = get_AUC(P_pre_probs, P_label)  # todo
        #fpr,tpr = get_roc(P_pre_probs,P_label,AUC,self.result_path)

        # save
        if (save_detail_result):
            # excel_save_path = os.path.join(self.result_path, mode + '_pre_detial_result.xlsx')
            excel_save_path = os.path.join(self.result_path, 'prediction', mode + '_' + str(current_epoch) + 'pre.xlsx')
            if not os.path.exists(os.path.join(self.result_path, 'prediction')):
                os.makedirs(os.path.join(self.result_path, 'prediction'))
            writer = pd.ExcelWriter(excel_save_path)
            detail_result1 = pd.DataFrame(detail_result1)
            detail_result1.to_excel(writer, 'patch_msg', float_format='%.5f')
            detail_result2 = pd.DataFrame(detail_result2)
            detail_result2.to_excel(writer, 'patient_msg', float_format='%.5f')
            detail_result3 = pd.DataFrame(np.array([accuracy, sensitivity, specificity, threshold, AUC]))
            detail_result3.to_excel(writer, 'patient_result', float_format='%.5f')
            writer.save()
            writer.close()
        # self.myprint('%s result has been Successfully Saved in %s' % (mode, excel_save_path))

        if during_Trianing:
            return accuracy, sensitivity, specificity, threshold, AUC, cost
        else:
            return accuracy, sensitivity, specificity, threshold, AUC

    def test_or(self, mode='train', classnet_path=None, save_detail_result=False, during_Trianing=False):
        if not classnet_path is None:
            if os.path.isfile(classnet_path):
                self.classnet.load_state_dict(torch.load(classnet_path))
                self.myprint('%s is Successfully Loaded from %s' % (self.model_type, classnet_path))

        self.classnet.train(False)
        self.classnet.eval()

        if mode == 'train':
            data_lodear = self.train_loader
        elif mode == 'test':
            data_lodear = self.test_loader
        elif mode == 'valid':
            data_lodear = self.valid_loader
        elif mode == 'extra':
            if self.extra_loader is None:
                print('Extra data is not existed!!')
                return
            data_lodear = self.extra_loader
        # else:
        # 	ValueError

        # model pre for each patch
        patient_order_list = []
        patch_order_list = []
        pre_list = []
        label_list = []
        cost = 0.0
        for i, sample in enumerate(data_lodear):
            (patch_paths, patch, label) = sample
            patch_paths = list(patch_paths)
            with torch.no_grad():
                patch = patch.to(self.device)
                label = label.to(self.device)
                # print(patch.shape)
                pre_probs = self.classnet(patch)
                pre_probs = F.sigmoid(pre_probs)#todo

                pre_flat = pre_probs.view(-1)
                label_flat = label.view(-1)
                loss = self.criterion(pre_flat, label_flat)
                cost += float(loss)

            pre_probs = pre_probs.data.cpu().numpy()
            label = label.data.cpu().numpy()

            for ii in range(pre_probs.shape[0]):
                pre_tmp = pre_probs[ii, :]
                label_tmp = label[ii, :]

                pre_list.append(pre_tmp.reshape(-1))
                label_list.append(label_tmp.reshape(-1))

                tmp_index = patch_paths[ii].split('/')[-1]
                tmp_index1 = tmp_index.split('_')[0][:]
                patient_order_list.append(int(tmp_index1))
                tmp_index2 = tmp_index.split('_')[1][:]
                tmp_index2 = tmp_index2.split('.')[0][:]
                patch_order_list.append(int(tmp_index2))

        cost /= (i + 1)

        detail_result1 = np.zeros([len(patient_order_list), 4])  # detail_msg = [id, patch_id, pre, label]
        detail_result1[:, 0] = np.array(patient_order_list).T
        detail_result1[:, 1] = np.array(patch_order_list).T
        detail_result1[:, 2] = np.array(pre_list).T
        detail_result1[:, 3] = np.array(label_list).T

        # statistic for each patient
        patinet_order_unique = np.unique(patient_order_list)
        detail_result2 = np.zeros([len(patinet_order_unique), 4])  # detail_msg = [id, _, mpre, label]
        detail_result2[:, 0] = patinet_order_unique.T
        for unique_p_order in patinet_order_unique:
            select_patient_index = [i for i, x in enumerate(patient_order_list) if x == unique_p_order]
            pre_tmp = []
            label_tmp = []
            for i in select_patient_index:
                pre_tmp.append(pre_list[i])
                label_tmp.append(label_list[i])
            pre_probs = np.array(pre_tmp).reshape(-1)
            label = np.array(label_tmp).reshape(-1)
            if mode == 'valid':
                mean_pre_probs = np.max(pre_probs)
            if mode == 'extra':
                mean_pre_probs = np.max(pre_probs)
            label = np.mean(label)

            # detail_result[detail_result[:,0] == unique_p_order,1] = get_AUC(pre_probs, label)
            detail_result2[detail_result2[:, 0] == unique_p_order, 2] = mean_pre_probs
            detail_result2[detail_result2[:, 0] == unique_p_order, 3] = label

        P_pre_probs = torch.from_numpy(detail_result2[:, 2]).to(self.device)
        P_label = torch.from_numpy(detail_result2[:, 3]).to(self.device)

        if mode == 'train':
            threshold = get_bset_threshold(P_pre_probs, P_label)
        elif mode == 'valid':
            threshold = get_bset_threshold(P_pre_probs, P_label)
            self.pre_threshold = threshold
        elif mode == 'extra':
            threshold = get_bset_threshold(P_pre_probs, P_label)
            self.pre_threshold = threshold
        else:
            threshold = self.pre_threshold
        accuracy = get_accuracy(P_pre_probs, P_label, threshold)
        sensitivity = get_sensitivity(P_pre_probs, P_label, threshold)
        specificity = get_specificity(P_pre_probs, P_label, threshold)
        AUC = get_AUC(P_pre_probs, P_label)  # todo
        fpr,tpr = get_roc(P_pre_probs,P_label,AUC,self.result_path)

        # save
        if (save_detail_result):
            excel_save_path = os.path.join(self.result_path, mode + '_pre_detial_result.xlsx')
            writer = pd.ExcelWriter(excel_save_path)
            detail_result1 = pd.DataFrame(detail_result1)
            detail_result1.to_excel(writer, 'patch_msg', float_format='%.5f')
            detail_result2 = pd.DataFrame(detail_result2)
            detail_result2.to_excel(writer, 'patient_msg', float_format='%.5f')
            detail_result3 = pd.DataFrame(np.array([accuracy, sensitivity, specificity, threshold, AUC]))
            detail_result3.to_excel(writer, 'patient_result', float_format='%.5f')
            writer.save()
            writer.close()
        # self.myprint('%s result has been Successfully Saved in %s' % (mode, excel_save_path))

        if during_Trianing:
            return accuracy, sensitivity, specificity, threshold, AUC, cost,fpr,tpr
        else:
            return accuracy, sensitivity, specificity, threshold, AUC,fpr,tpr

    def test_cam(self, mode='train', classnet_path=None, save_detail_result=False, during_Trianing=False):
        """Test model & Calculate performances."""
        if not classnet_path is None:
            if os.path.isfile(classnet_path):
                self.classnet.load_state_dict(torch.load(classnet_path))
                self.myprint('%s is Successfully Loaded from %s' % (self.model_type, classnet_path))
        layer = list(list(list(self.classnet.children())[-3].children())[-1].children())[-3]
        self.classnet = gcam.inject(self.classnet, output_dir=self.result_path, save_maps=True, backend='gcampp',
                                    layer='layer3.2.conv3', data_shape='(16,256,256)')

        self.classnet.train(True)
        self.classnet.eval()

        if mode == 'train':
            data_lodear = self.train_loader
        elif mode == 'test':
            data_lodear = self.test_loader
        elif mode == 'valid':
            data_lodear = self.valid_loader
        elif mode == 'extra':
            if self.extra_loader is None:
                print('Extra data is not existed!!')
                return
            data_lodear = self.extra_loader
        # else:
        # 	ValueError

        # model pre for each patch
        patient_order_list = []
        patch_order_list = []
        pre_list = []
        label_list = []
        cost = 0.0
        for i, sample in enumerate(data_lodear):
            (patch_paths, patch, label) = sample
            patch_paths = list(patch_paths)
            patch = patch.to(self.device)
            label = label.to(self.device)
            pre_probs = self.classnet(patch)
            # pre_probs = F.sigmoid(pre_probs)#todo

            pre_flat = pre_probs.view(-1)
        # 	label_flat = label.view(-1)
        # 	loss = self.criterion(pre_flat,label_flat)
        # 	cost += float(loss)
        #
        # 	pre_probs = pre_probs.data.cpu().numpy()
        # 	label = label.data.cpu().numpy()
        #
        # 	for ii in range(pre_probs.shape[0]):
        # 		pre_tmp = pre_probs[ii,:]
        # 		label_tmp = label[ii, :]
        #
        # 		pre_list.append(pre_tmp.reshape(-1))
        # 		label_list.append(label_tmp.reshape(-1))
        #
        # 		tmp_index = patch_paths[ii].split('/')[-1]
        # 		tmp_index1 = tmp_index.split('_')[0][:]
        # 		patient_order_list.append(int(tmp_index1))
        # 		tmp_index2 = tmp_index.split('_')[1][:]
        # 		tmp_index2 = tmp_index2.split('.')[0][:]
        # 		patch_order_list.append(int(tmp_index2))
        #
        # cost /= (i+1)
        #
        # detail_result1 = np.zeros([len(patient_order_list), 4])  # detail_msg = [id, patch_id, pre, label]
        # detail_result1[:, 0] = np.array(patient_order_list).T
        # detail_result1[:, 1] = np.array(patch_order_list).T
        # detail_result1[:, 2] = np.array(pre_list).T
        # detail_result1[:, 3] = np.array(label_list).T
        #
        # # statistic for each patient
        # patinet_order_unique = np.unique(patient_order_list)
        # detail_result2 = np.zeros([len(patinet_order_unique),4]) # detail_msg = [id, _, mpre, label]
        # detail_result2[:,0] = patinet_order_unique.T
        # for unique_p_order in patinet_order_unique:
        # 	select_patient_index = [i for i, x in enumerate(patient_order_list) if x == unique_p_order]
        # 	pre_tmp = []
        # 	label_tmp = []
        # 	for i in select_patient_index:
        # 		pre_tmp.append(pre_list[i])
        # 		label_tmp.append(label_list[i])
        # 	pre_probs = np.array(pre_tmp).reshape(-1)
        # 	label = np.array(label_tmp).reshape(-1)
        # 	mean_pre_probs = np.mean(pre_probs) # TODO
        # 	label = np.mean(label)
        #
        # 	# detail_result[detail_result[:,0] == unique_p_order,1] = get_AUC(pre_probs, label)
        # 	detail_result2[detail_result2[:,0] == unique_p_order, 2] = mean_pre_probs
        # 	detail_result2[detail_result2[:,0] == unique_p_order, 3] = label
        #
        # P_pre_probs = torch.from_numpy(detail_result2[:,2]).to(self.device)
        # P_label = torch.from_numpy(detail_result2[:,3]).to(self.device)
        #
        # if  mode=='train':
        # 	threshold = get_bset_threshold(P_pre_probs, P_label)
        # elif  mode=='valid':
        # 	threshold = get_bset_threshold(P_pre_probs, P_label)
        # 	self.pre_threshold = threshold
        # else:
        # 	threshold = self.pre_threshold
        # accuracy = get_accuracy(P_pre_probs, P_label,threshold)
        # sensitivity = get_sensitivity(P_pre_probs, P_label,threshold)
        # specificity = get_specificity(P_pre_probs, P_label,threshold)
        # AUC = get_AUC(P_pre_probs, P_label) #todo
        #
        # # save
        # if(save_detail_result):
        # 	excel_save_path = os.path.join(self.result_path, mode + '_pre_detial_result.xlsx')
        # 	writer = pd.ExcelWriter(excel_save_path)
        # 	detail_result1 = pd.DataFrame(detail_result1)
        # 	detail_result1.to_excel(writer, 'patch_msg', float_format='%.5f')
        # 	detail_result2 = pd.DataFrame(detail_result2)
        # 	detail_result2.to_excel(writer, 'patient_msg', float_format='%.5f')
        # 	detail_result3 = pd.DataFrame(np.array([accuracy, sensitivity, specificity, threshold, AUC]))
        # 	detail_result3.to_excel(writer, 'patient_result', float_format='%.5f')
        # 	writer.save()
        # 	writer.close()
        # 	# self.myprint('%s result has been Successfully Saved in %s' % (mode, excel_save_path))
        #
        # if during_Trianing:
        # 	return accuracy, sensitivity, specificity,threshold, AUC, cost
        # else:
        # 	return accuracy, sensitivity, specificity,threshold, AUC



