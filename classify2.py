import argparse
import os
# from solver_withLinchuang import Solver
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from my_process_funtion import get_fold_filelist
import random
from evaluation import *

aucs = []
fprs = []
tprs = []
def main(config):
    cudnn.benchmark = True

    config.Task_name = config.Task_name + '_' + str(config.fold_idx) + '_' + str(config.fold_K)
    config.result_path = os.path.join(config.result_path, config.Task_name)
    config.model_path = os.path.join(config.result_path, 'models')
    config.log_dir = os.path.join(config.result_path, 'logs')
    # Create directories if not exist
    if config.mode == 'train':
        if not os.path.exists(config.result_path):
            os.makedirs(config.result_path)
            os.makedirs(config.model_path)
            os.makedirs(config.log_dir)

        print(config)
        f = open(os.path.join(config.result_path, 'config.txt'), 'w')
        for key in config.__dict__:
            print('%s: %s' % (key, config.__getattribute__(key)), file=f)
        f.close()

    if config.mode == 'test_cam':
        config.batch_size = 1
        config.batch_size_test = 1
        config.augmentation_prob = 0.
    if not config.DataParallel:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_idx)

    # make_cross_fold
    '分折'
    # data_all_list = os.listdir(config.h5data_path)
    #
    # data_all_list = list(int(i) for i in data_all_list)
    # data_all_list.sort()
    #
    #
    # train_set, test_set = get_fold_filelist(data_all_list, config.label_csv_file,
    #                                         K=config.fold_K, fold=config.fold_idx,
    #                                         validation=False)
    #
    # # train_list = train_set_all[config.fold_idx-1]
    # # test_list = test_set_all[config.fold_idx-1]
    # valid_set = test_set
    # name_base = ''
    # train_list = [name_base[:-len(str(i[0]))] + str(i[0]) for i in train_set]
    # valid_list = [name_base[:-len(str(i[0]))] + str(i[0]) for i in valid_set]
    # test_list = [name_base[:-len(str(i[0]))] + str(i[0]) for i in test_set]
    #
    #
    # config.train_list = train_list
    # config.valid_list = valid_list
    # config.test_list = config.valid_list
    # config.test_list = test_list

    lable_list = np.loadtxt(open(config.label_csv_file, "rb"), delimiter=",", skiprows=0)
    lable_list = np.array(lable_list, dtype='int16')
    data_all_list = os.listdir(config.h5data_path)
    data_all_list = list(int(i) for i in data_all_list)
    data_all_list.sort()
    data_list = [i for i in lable_list if i[0] in data_all_list]
    data_list = np.array(data_list, dtype='int16')
    train_set = data_list
    name_base = ''
    train_list = [name_base[:-len(str(i[0]))] + str(i[0]) for i in train_set]
    valid_set = train_set
    test_set = train_set
    valid_list = [name_base[:-len(str(i[0]))] + str(i[0]) for i in valid_set]
    test_list = [name_base[:-len(str(i[0]))] + str(i[0]) for i in test_set]
    config.train_list = train_list
    config.valid_list = valid_list
    config.test_list = test_list

    train_loader = get_loader(image_root=config.h5data_path,
                              image_list=train_list,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_root=config.h5data_path,
                              image_list=valid_list,
                              image_size=config.image_size,
                              batch_size=config.batch_size_test,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0.)
    test_loader = get_loader(image_root=config.h5data_path,
                             image_list=test_list,
                             image_size=config.image_size,
                             batch_size=config.batch_size_test,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0.)

    if config.with_extra_data:
        extra_list = os.listdir(config.extra_h5data_path)
        extra_list.sort(key = lambda x:int(x[:]))
        config.extra_list = extra_list
        extra_loader = get_loader(image_root=config.extra_h5data_path,
                                  image_list=extra_list,
                                  image_size=config.image_size,
                                  batch_size=config.batch_size_test,
                                  num_workers=config.num_workers,
                                  mode='extra',
                                  augmentation_prob=0.)

        extra_list2 = os.listdir(config.extra_h5data_path2)
        extra_list2.sort(key = lambda x:int(x[:]))
        config.extra_list2 = extra_list2
        extra_loader2 = get_loader(image_root=config.extra_h5data_path2,
                                  image_list=extra_list2,
                                  image_size=config.image_size,
                                  batch_size=config.batch_size_test,
                                  num_workers=config.num_workers,
                                  mode='extra',
                                  augmentation_prob=0.)
    else:
        config.extra_list = []
        extra_loader = None

    solver = Solver(config, train_loader, valid_loader, test_loader, extra_loader,extra_loader2)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':

        mytest = solver.test
        # acc, SE, SP, threshold, AUC = mytest(mode='train', classnet_path=classnet_path, save_detail_result=True)
        # print('[Training  ] Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (acc, SE, SP, threshold, AUC))
        # acc, SE, SP, threshold, AUC = mytest(mode='valid', classnet_path=classnet_path, save_detail_result=True)
        # print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (acc, SE, SP, threshold, AUC))
        # acc, SE, SP, threshold, AUC= mytest(mode='test', classnet_path=classnet_path, save_detail_result=True)
    elif config.mode == 'test_or':
        model_list = os.listdir('/data/LHX/bone/model_new')
        #model_list = os.listdir('/home/szu/ray/Ray/models_roc')
        model_list.sort(key=lambda x: (x[0]))
        model_name = model_list[config.fold_idx-1]
        print(model_name)
        classnet_path = os.path.join('/data/LHX/bone/model_new',model_name)
        print(classnet_path)
        #classnet_path = os.path.join('/home/szu/ray/Ray/models_roc', model_name)
        mytest = solver.test_or
        acc, SE, SP, threshold, AUC,fpr,tpr = mytest(mode='extra', classnet_path=classnet_path, save_detail_result=True)
        aucs.append(AUC)
        tprs.append(tpr)
        fprs.append(fpr)
        # tprs.extend(tpr)
        # fprs.extend(fpr)
        print('[Testing]    Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (acc, SE, SP, threshold, AUC))
        # acc, SE, SP, threshold, AUC = mytest(mode='extra', classnet_path=classnet_path, save_detail_result=True)
        # print(
        #     '[Extra]        Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (acc, SE, SP, threshold, AUC))
    elif config.mode == 'test_cam':
        classnet_path = os.path.join(config.model_path, 'epoch980_Testdice0.6429.pkl')
        mytest = solver.test_cam
        mytest(mode='valid', classnet_path=classnet_path, save_detail_result=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--patch_size', type=int, default=12)

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--batch_size_test', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=16)  # !!!dont change!!!
    parser.add_argument('--lr', type=float, default=0.001)




    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)
    parser.add_argument('--num_epochs_decay', type=int, default=100)  # decay开始的最小epoch数
    parser.add_argument('--decay_ratio', type=float, default=0.9)  # 0~1,每次decay到1*ratio
    parser.add_argument('--decay_step', type=int, default=20)  # epoch
    parser.add_argument('--lr_low', type=float, default=1e-5)  # 最小学习率,设置为None,则为最大学习率的1e+6分之一(不可设置为0)
    parser.add_argument('--lr_warm_epoch', type=int, default=20)  # warmup的epoch数,一般就是10~20,为0或False则不使用
    parser.add_argument('--lr_cos_epoch', type=int, default=100)  # cos退火的epoch数,一般就是总epoch数-warmup的数,为0或False则代表不使用

    # misc  n
    parser.add_argument('--mode', type=str, default='train', help='train/test/test_cam')
    parser.add_argument('--model_type', type=str, default='efficientnet',
                        help='resnet/densenet/sparsenet/seresnet/resnet50C2D/my_NLresnet')
    parser.add_argument('--Task_name', type=str, default='CAM_test', help='DIR name,Task name')  # arterial venous
    # parser.add_argument('--Task_name', type=str, default='venous_fold1_new', help='DIR name,Task name') # arterial venous
    parser.add_argument('--cuda_idx', type=int, default=1)
    parser.add_argument('--DataParallel', type=bool, default=False)

    # data-parameters


    parser.add_argument('--h5data_path', type=str,
                        default='/data/LHX/graduate/labled_data_50%_alpha0.1')  # arterial venous
    parser.add_argument('--label_csv_file', type=str,
                        default='/data/LHX/bone/label_all.csv')  # arterial venous
    parser.add_argument('--with_extra_data', type=bool, default=True)
    # parser.add_argument('--model_path',type=str,default='/home/szu/liver/final_nii_data/result/CAM_test_1_5')
    parser.add_argument('--extra_h5data_path', type=str,
                        default='/data/LHX/bone/test_all')  # arterial venous
    parser.add_argument('--extra_h5data_path2', type=str,
                        default='/data/LHX/bone/extra_all')


    parser.add_argument('--fold_K', type=int, default=1, help='folds number after divided')
    parser.add_argument('--fold_idx', type=int, default=1)
    # parser.add_argument('--train_path', type=str, default='./dataset/train/')
    # parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    # parser.add_argument('--test_path', type=str, default='./dataset/test/')

    # result&save
    parser.add_argument('--result_path', type=str, default='/data/LHX/graduate/50%labled_alpha0.1_AL_data')
    parser.add_argument('--save_detail_result', type=bool, default=True)

    config = parser.parse_args()

    main(config)

    'K折交叉验证'
    # for i in range(0,5):
    #     parser = argparse.ArgumentParser()
    #
    #     # model hyper-parameters
    #     parser.add_argument('--image_size', type=int, default=512)
    #     parser.add_argument('--patch_size', type=int, default=12)
    #
    #     # training hyper-parameters
    #     parser.add_argument('--img_ch', type=int, default=1)
    #     parser.add_argument('--output_ch', type=int, default=1)
    #     parser.add_argument('--num_epochs', type=int, default=150)
    #     parser.add_argument('--batch_size', type=int, default=4)
    #     parser.add_argument('--batch_size_test', type=int, default=2)
    #     parser.add_argument('--num_workers', type=int, default=16)  # !!!dont change!!!
    #     parser.add_argument('--lr', type=float, default=0.001)
    #
    #
    #
    #
    #     parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    #     parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    #     parser.add_argument('--augmentation_prob', type=float, default=0)
    #
    #     parser.add_argument('--log_step', type=int, default=2)
    #     parser.add_argument('--val_step', type=int, default=2)
    #     parser.add_argument('--num_epochs_decay', type=int, default=100)  # decay开始的最小epoch数
    #     parser.add_argument('--decay_ratio', type=float, default=0.9)  # 0~1,每次decay到1*ratio
    #     parser.add_argument('--decay_step', type=int, default=20)  # epoch
    #     parser.add_argument('--lr_low', type=float, default=1e-5)  # 最小学习率,设置为None,则为最大学习率的1e+6分之一(不可设置为0)
    #     parser.add_argument('--lr_warm_epoch', type=int, default=20)  # warmup的epoch数,一般就是10~20,为0或False则不使用
    #     parser.add_argument('--lr_cos_epoch', type=int, default=100)  # cos退火的epoch数,一般就是总epoch数-warmup的数,为0或False则代表不使用
    #
    #     # misc  n
    #     parser.add_argument('--mode', type=str, default='train', help='train/test/test_cam')
    #     parser.add_argument('--model_type', type=str, default='efficientnet',
    #                         help='resnet/densenet/sparsenet/seresnet/resnet50C2D/my_NLresnet')
    #     parser.add_argument('--Task_name', type=str, default='CAM_test', help='DIR name,Task name')  # arterial venous
    #     # parser.add_argument('--Task_name', type=str, default='venous_fold1_new', help='DIR name,Task name') # arterial venous
    #     parser.add_argument('--cuda_idx', type=int, default=2)
    #     parser.add_argument('--DataParallel', type=bool, default=False)
    #
    #     # data-parameters
    #
    #
    #     parser.add_argument('--h5data_path', type=str,
    #                         default='/data/LHX/graduate/labled_data')  # arterial venous
    #     parser.add_argument('--label_csv_file', type=str,
    #                         default='/data/LHX/bone/label_all.csv')  # arterial venous
    #     parser.add_argument('--with_extra_data', type=bool, default=True)
    #     # parser.add_argument('--model_path',type=str,default='/home/szu/liver/final_nii_data/result/CAM_test_1_5')
    #     parser.add_argument('--extra_h5data_path', type=str,
    #                         default='/data/LHX/bone/test_all')  # arterial venous
    #     parser.add_argument('--extra_h5data_path2', type=str,
    #                         default='/data/LHX/bone/extra_all')
    #
    #
    #     parser.add_argument('--fold_K', type=int, default=5, help='folds number after divided')
    #     parser.add_argument('--fold_idx', type=int, default=i + 1)
    #     # parser.add_argument('--train_path', type=str, default='./dataset/train/')
    #     # parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    #     # parser.add_argument('--test_path', type=str, default='./dataset/test/')
    #
    #     # result&save
    #     parser.add_argument('--result_path', type=str, default='/data/LHX/graduate/40%labled_data')
    #     parser.add_argument('--save_detail_result', type=bool, default=True)
    #
    #     config = parser.parse_args()
    #
    #     main(config)

    #get_roc_inside(fprs,tprs,path = '/home/szu/liver/result_arterial')
    # get_roc_plot_extra(aucs,fprs,tprs,path = '/home/szu/bladder/model')
