from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from model.pytorch.supervisor import LSCGFSupervisor

parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', default='data/model/para-bay', type=str)
parser.add_argument('--temperature', default=0.5, type=float, help='temperature value for gumbel-softmax.')
# basic settings
parser.add_argument('--device',default='cuda:1',type=str)
parser.add_argument('--log_dir',default='data/model',type=str,help='')
parser.add_argument('--log_level',default='INFO',type=str)
parser.add_argument('--log_every',default=1,type=int)
parser.add_argument('--save_model',default=0,type=int)
#data settings
parser.add_argument('--batch_size',default=64,type=int)
parser.add_argument('--dataset_dir',default='data/METR-LA',type=str)
parser.add_argument('--test_batch_size',default=64,type=int)
parser.add_argument('--valid_batch_size',default=64,type=int)
# model settings
parser.add_argument('--cl_decay_steps',default=2000,type=int)
parser.add_argument('--filter_type',default='dual_random_walk',type=str)
parser.add_argument('--horizon',default=12,type=int)
parser.add_argument('--input_dim',default=2,type=int)
parser.add_argument('--ll_decay',default=0,type=int)
parser.add_argument('--max_diffusion_step',default=2,type=int)
parser.add_argument('--num_rnn_layers',default=1,type=int)
parser.add_argument('--output_dim',default=1,type=int)
parser.add_argument('--rnn_units',default=64,type=int)
parser.add_argument('--seq_len',default=12,type=int)
parser.add_argument('--use_curriculum_learning',default=True,type=bool)
parser.add_argument('--embedding_size',default=256,type=int)
parser.add_argument('--kernel_size',default=12,type=int)
parser.add_argument('--freq',default=288,type=int)
parser.add_argument('--requires_graph',default=2,type=int)
# train settings0
parser.add_argument('--base_lr',default=0.003,type=float)
parser.add_argument('--dropout',default=0.3,type=float)
parser.add_argument('--epoch',default=0,type=int)
parser.add_argument('--epochs',default=200,type=int)
parser.add_argument('--epsilon',default=1.0e-3,type=float)
parser.add_argument('--global_step',default=0,type=int)
parser.add_argument('--lr_decay_ratio',default=0.1,type=float)
parser.add_argument('--max_grad_norm',default=5,type=int)
parser.add_argument('--max_to_keep',default=100,type=int)
parser.add_argument('--min_learning_rate',default=2.0e-05,type=float)
parser.add_argument('--optimizer',default='adam',type=str)
parser.add_argument('--patience',default=50,type=int)
parser.add_argument('--steps',default=[20, 30, 40],type=list)
parser.add_argument('--test_every_n_epochs', default=5, type=int)
parser.add_argument('--num_sample', default=10, type=int)
args = parser.parse_args()


if __name__ == '__main__':
    print(args)
    save_adj_name = args.config_filename[11:-5]
    supervisor = LSCGFSupervisor(save_adj_name, args=args)
    supervisor.train(args)
