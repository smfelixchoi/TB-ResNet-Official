import argparse, glob, os, torch, warnings, time

from utils.tools import *
from utils.dataLoader import train_loader
from TBResNetModel import TBResNetModel

parser = argparse.ArgumentParser(description = "TResNet_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=128,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.0001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="/home/ubuntu/Data/VoxCeleb/VoxCeleb2/train_list_299.txt",     
                    help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')

parser.add_argument('--train_path', type=str,   default="/home/ubuntu/Data/VoxCeleb/ECAPATDNN/sample_data",     
                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')

parser.add_argument('--eval_list',  type=str,   default="/home/ubuntu/Data/VoxCeleb/veri_test2.txt",          
                    help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')

parser.add_argument('--eval_path',  type=str,   default="/home/ubuntu/Data/VoxCeleb/VoxCeleb1",             
                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')

parser.add_argument('--musan_path', type=str,   default="/home/ubuntu/Data/musan_split",                
                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')

parser.add_argument('--rir_path',   type=str,   default="/home/ubuntu/Data/RIRS_NOISES/simulated_rirs",  
   help='The path to the RIR set, eg:"/home/ubuntu/Data/RIRS_NOISES/simulated_rirs" in my case');

parser.add_argument('--save_path',  type=str,   default="/home/ubuntu/user_MOOK/SASV/CNN-based-torch/T_ResNet18_2/exps/299_samples_StatisticsPooling_channel_mu_std",        
                    help='Path to save the score.txt and models')

parser.add_argument('--initial_model',  type=str,   default="",                               
                    help='Path of the initial_model')


## Model and Loss settings
# parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--attention',type=str,  default='channel',help='Attention Type')
parser.add_argument('--stats',    type=str,  default='mu_std', help='Statistics for Pooling')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=299,    help='Number of speakers')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

## Define the data loader
trainloader = train_loader(**vars(args))
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

## Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = TBResNetModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	EER, minDCF = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
	print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = TBResNetModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = TBResNetModel(**vars(args))
	s.load_parameters(modelfiles[-1])
## Otherwise, system will train from scratch
else:
	epoch = 1
	s = TBResNetModel(**vars(args))

EERs = []
DCFs = []
score_file = open(args.score_save_path, "a+")

while(1):
    ## Training for one epoch
    loss, lr, acc = s.train_network(epoch = epoch, loader = trainLoader)

    ## Evaluation every [test_step] epochs
    if epoch % args.test_step == 0:
        s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
        temp_EER, temp_minDCF = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
        EERs.append(temp_EER)
        DCFs.append(temp_minDCF)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, minDCF %.4f%%, bestEER %2.2f%%, bestminDCF %.4f%%"%(epoch, acc, EERs[-1], DCFs[-1], min(EERs), min(DCFs)))
        score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, minDCF %.4f%%, bestEER %2.2f%%, bestminDCF %.4f%%\n"%(epoch, lr, loss, acc, EERs[-1], DCFs[-1], min(EERs), min(DCFs)))
        score_file.flush()

    if epoch >= args.max_epoch:
        quit()

    epoch += 1
