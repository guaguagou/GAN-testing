import pandas as pd
import numpy as np
import os
from random import sample
num_inputs = 3
num_pt = 96
pt_start = 295
data = ['/Users/guagou/PycharmProjects/GAN-testing/dataset/convWGAN-GP/path_state/Hyperg/F18/Hy1F1_gsl_sf_hyperg_1F1_F18.csv']
GAN_op_file = ['/Users/guagou/PycharmProjects/GAN-testing/dataset/convWGAN-GP/path_state/Hyperg/F18/']


i = 0
while i < 10:
    train_data = pd.read_csv(r'{}'.format(data[0]))
    batch_size = train_data.shape[0]
    column_name = train_data.columns.tolist()
    #print(train_data)
    #get uncovered branch index
    ii = pt_start
    uncover_index = []
    while ii < pt_start+num_pt:
        reqd_Index = train_data[train_data['b{}'.format(ii)]==1.5].index.tolist()
        if reqd_Index:
            ii=ii+1
        else:
            # The list is empty
            uncover_index.append(ii)
            ii = ii + 1

    if i==0:
        cover = (num_pt-len(uncover_index))/num_pt
        f = open('{}cover.txt'.format(GAN_op_file[0]), 'a')
        f.write('uncover branchs: {}     initial coverage rate is {} ,'.format(uncover_index, cover))
        f.close()

    GAN_out = '{}{}.csv'.format(GAN_op_file[0], i)
    #train GAN
    #os.system('python WGAN-GPHy.py data[0], num_inputs, num_pt, batch_size, GAN_out, i')
    os.system("python convWGAN-GP.py %s %d %d %d %s %d %s" % (data[0],num_inputs,num_pt,batch_size,GAN_out,i,GAN_op_file[0]))

    #select test cases based on predict branch information
    fake_data = pd.read_csv(r'{}'.format(GAN_out), header=None)
    fake_data.columns = column_name

    iii=0
    b_argmax =([])
    while iii < len(uncover_index):
        column= fake_data.loc[:,'b{}'.format(uncover_index[iii])]
        #argmax = column.idxmax()
        b_argmax.append(column[column == 1.0].index.values)
        iii=iii+1

    if len(b_argmax)>0:
        #flatten
        b_argmax1 = [x for y in b_argmax for x in y]
        #remove the same
        b_argmax2 = list(set(b_argmax1))
    else:
        #random select 10 sample index
        b_argmax2 = range(1, 10)
    #select 20 sample
    if len(b_argmax2)>10:
        select_index = sample(b_argmax2, 10)
    else:
        select_index = b_argmax2
    N = len(select_index)
    select_input = fake_data[column_name[:3]].loc[select_index]
    np.savetxt('/Users/guagou/Desktop/code/test_block/hyperg_1F1GAN/hyperg_1F1Test1/10_100.csv', select_input, delimiter=',')

    #execute program under test
    x = 0
    while x < N:
        cmd = "cd /Users/guagou/Desktop/code/test_block/hyperg_1F1GAN/hyperg_1F1Test1/ && " \
              "gcc --coverage hyperg_1F1.c -c && " \
              "gcc --coverage hyperg_1F1.o -lgsl -lgslcblas && " \
              "./a.out {} && " \
              "gcov hyperg_1F1.c -b -c && " \
              "mv hyperg_1F1.c.gcov {}.txt && " \
              "mv {}.txt /Users/guagou/PycharmProjects/GAN-testing/temporar_path && " \
              "rm hyperg_1F1.o a.out hyperg_1F1.gcno hyperg_1F1.gcda".format(x, x, x)
        os.system(cmd)
        x = x+1

    cmd1 = "python path_extract_state.py {}".format(N)
    a = os.system(cmd1)
    real_all_path = pd.read_csv(r'extract_path.csv', header=None)
    real_func_path = real_all_path.iloc[:,pt_start-1:pt_start+num_pt-1].values
    real_input_path = np.concatenate((select_input, real_func_path), axis=1)

#remove invalid data (can not execute the func under test)
    real_input_path1 = pd.DataFrame(real_input_path)
    oneth_branch = real_input_path1.loc[:, num_inputs]
    invalid_index = oneth_branch[oneth_branch == 2.5].index.values
    if invalid_index.size > 0:
        valid_input_path = np.delete(real_input_path,invalid_index, axis=0)
    else:
        valid_input_path = real_input_path

    if len(valid_input_path)>0:
        valid_input_path = pd.DataFrame(valid_input_path)
        valid_input_path.columns = column_name
        old_train = pd.read_csv(r'{}'.format(data[0]))
        new_train = train_data.append(valid_input_path)
        new_train.to_csv(data[0], index=0)
    i=i+1

ii = pt_start
uncover_index = []
while ii < pt_start+num_pt:
    reqd_Index = train_data[train_data['b{}'.format(ii)]==1.5].index.tolist()
    if reqd_Index:
        ii=ii+1
    else:
            # The list is empty
        uncover_index.append(ii)
        ii = ii + 1
print(uncover_index)

cover_rate = (num_pt-len(uncover_index))/num_pt
f = open('{}cover.txt'.format(GAN_op_file[0]), 'a')
f.write('uncover branchs: {}     final coverage rate is {} ,'.format(uncover_index, cover_rate))
f.close()
