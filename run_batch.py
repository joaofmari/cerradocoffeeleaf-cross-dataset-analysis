# Author: João Fernando Mari
# joaofmari.github.io
# https://github.com/joaofmari

import os
import argparse
import shutil

# Available datasets and respective splits:
# ----------------------------------------
# ['BRACOL',   'JMuBEN',   'RoCoLe',   'CerradoCoffeLeaf']
# ['no-split', 'no-split', 'no-split', 'train-val-test']

split_dict = {'BRACOL' : 'no-split',
              'JMuBEN' : 'no-split',
              'RoCoLe' : 'no-split',
              'CerradoCoffeLeaf' : 'train-val-test',
              }

parser = argparse.ArgumentParser()

parser.add_argument('--exp', help='Select the experiment setup: [1, 2, 3, 4].', type=int, default=1)

# Processes the arguments provided on the command line
args = parser.parse_args()

if args.exp == 1:
    print('>> Running EXP 1: Train in CerradoCoffeLeaf. Test in OTHERS.')
    # **** EXP 1 *****
    # Train in CerradoCoffeLeaf. Test in OTHERS.
    # ---------------------------------------------------------
    ds_list =       ['CerradoCoffeLeaf', 'CerradoCoffeLeaf', 'CerradoCoffeLeaf', 'CerradoCoffeLeaf']
    ds1_list =      ['none',             'none',             'none',             'none']

    ds_test_list =  ['CerradoCoffeLeaf', 'JMuBEN',        'BRACOL',        'RoCoLe']
    ds1_test_list = ['none',             'none',          'none',          'none']

elif args.exp == 2:
    print('>> Running EXP 2: Train in OTHERS. Test in CerradoCoffeLeaf.')
    # **** EXP 2 *****
    # Train in OTHERS. Test in CerradoCoffeLeaf
    # ---------------------------------------------------------
    ds_list =       [ 'JMuBEN',        'BRACOL',        'RoCoLe']
    ds1_list =      [ 'none',          'none',          'none']

    ds_test_list =  [ 'CerradoCoffeLeaf', 'CerradoCoffeLeaf', 'CerradoCoffeLeaf']
    ds1_test_list = [ 'none',             'none',             'none']

elif args.exp == 3:
    print('>> Running EXP 3: Train in CerradoCoffeLeaf + Other. Test in CerradoCoffeLeaf, Other, and (CerradoCoffeLeaf + Other).')
    # **** EXP 3 *****
    # Train in CerradoCoffeLeaf + Other). Test in CerradoCoffeLeaf, Other, and (CerradoCoffeLeaf + Other).
    # ---------------------------------------------------------
    ds_list =      ['CerradoCoffeLeaf', 'CerradoCoffeLeaf', 'CerradoCoffeLeaf',
                    'CerradoCoffeLeaf', 'CerradoCoffeLeaf', 'CerradoCoffeLeaf',
                    'CerradoCoffeLeaf', 'CerradoCoffeLeaf', 'CerradoCoffeLeaf']
    ds1_list =      ['JMuBEN',        'BRACOL',        'RoCoLe',
                     'JMuBEN',        'BRACOL',        'RoCoLe',
                     'JMuBEN',        'BRACOL',        'RoCoLe']
    ds_test_list = ['CerradoCoffeLeaf', 'CerradoCoffeLeaf', 'CerradoCoffeLeaf',
                    'JMuBEN',           'BRACOL',           'RoCoLe',
                    'CerradoCoffeLeaf', 'CerradoCoffeLeaf', 'CerradoCoffeLeaf']
    ds1_test_list = ['none',            'none',             'none',
                     'none',            'none',             'none',
                     'JMuBEN',          'BRACOL',           'RoCoLe']

elif args.exp == 4:
    print('>> Running EXP 4: Train in OTHERS. Test in the same OTHERS.')
    # **** EXP 4 *****
    # Train in OTHERS. Test in the same OTHERS
    # ---------------------------------------------------------
    ds_list =       [ 'JMuBEN',        'BRACOL',        'RoCoLe']
    ds1_list =      [ 'none',          'none',          'none']

    ds_test_list =  [ 'JMuBEN',        'BRACOL',        'RoCoLe']
    ds1_test_list = [ 'none',          'none',          'none']

    EXP_PATH_MAIN = f'exp'

else:
    print('>> Invalid experiment setup. Please select a valid option: [1, 2, 3, 4].')
    exit(1)

# TTA strategy
dapred = 0

arch_list = ['resnet50', 
             'vit_b_16', 
             'swin_v2_t',
             'efficientnet_v2_s',
            # 'convnext_tiny'
            ]

# Data augmentation strategies. See the file 'data_aug_3.py'
# (The lists must have the same length.)
datrain_list = [1, 12] # [1, 11, 12]  # [1, 11, 12, 13, 14] 
daval_list = [1, 1, 1]
datest_list = [1, 1, 1]

# Total number of training epochs.
epochs = 200 # 200
num_workers = 0 # To ensure reproducibility. Change to a higher value for faster data loading.

# Set the batch size and learning rate for each architecture.
bs_dict = {'resnet50' : 64, 
           'vit_b_16' : 64, 
           'swin_v2_t' : 64,
           'efficientnet_v2_s' : 64,
           'convnext_tiny' : 64,
           } 

lr_dict = {'resnet50' : 0.0001, 
           'vit_b_16' : 0.0001,
           'swin_v2_t' : 0.0001,
           'efficientnet_v2_s' : 0.0001,
           'convnext_tiny' : 0.0001,
           } 

# Learning rate scheduler
scheduler = 'plateau'

# Consider only overlapping classes between ds and ds1.
overlap = '--overlap' # '--overlap', '--no-overlap'

# Experiment counter
ec = 0

# Loop over architectures, data augmentation strategies, and datasets.
for arch in arch_list:
    
    for datrain, daval, datest in zip(datrain_list, daval_list, datest_list):
    
        for ds_train, ds1_train, ds_test, ds1_test in zip(ds_list, ds1_list, ds_test_list, ds1_test_list):

            cmd_str = f'nohup python train_model.py --ds {ds_train} --ds_split {split_dict[ds_train]} ' + \
                      f'--ds_test {ds_test} --ds_test_split {split_dict[ds_test]} ' + \
                      f'--ds1 {ds1_train} --ds1_split {split_dict[ds1_train]} ' + \
                      f'--ds1_test {ds1_test} --ds1_test_split {split_dict[ds1_test]} {overlap} ' + \
                      f'--arch {arch} --num_workers {num_workers} ' + \
                      f'--scheduler {scheduler} --ss {0} ' + \
                      f'--bs {bs_dict[arch]} --lr {lr_dict[arch]} --ep {epochs} --optimizer Adam ' + \
                      f'--datrain {datrain} --daval {daval} --datest {datest} --ec {ec} '  

            print(cmd_str) # I don't know why, but we need to print the command string to avoid some execution errors. :)
            ec = ec + 1

            os.system(cmd_str)

# Store the nohup.out file in the experiment folder, if it exists. Used to further analyze the results and check for any errors during the execution of the experiments.
if os.path.exists('./nohup.out'):
    suffix = ''
    while True:
        if os.path.exists(os.path.join(EXP_PATH_MAIN, 'nohup' + suffix + '.out')):
            suffix += '_'
        else:
            break
    shutil.move('./nohup.out', os.path.join(EXP_PATH_MAIN, 'nohup' + suffix + '.out'))

print('Done! (run_batch)')