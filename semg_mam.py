import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import time
import numpy as np
import pandas as pd
from scipy import signal

from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data_process
from system_para import window_size,number_of_classes
from multi_attention import CNNMAM
import random

import copy

usegpu = torch.cuda.is_available()

#设置路径参数
path = 'data_x'
user_names = ['S%d'%x for x in range(1,11)]
# user_name = 'S1'
for loop in range(1):
    slices_acc=[]
    intach_acc=[]
    session_index = np.load('random_index_%d.npy'%loop)#随机session，需自己生成
    for user_name in user_names:
        # session_index = random.sample(range(0,6),6)
        train_index = session_index[0:3]
        val_index = session_index[3:4]
        test_index = session_index[4:]

        # data = []
        # labels = []
        user_path = path + '/' + str(user_name) + '/'
        file_names = os.listdir(str(user_path))
        file_paths = [user_path + i for i in file_names]

        user_data = None
        user_labels = []

        for txt_path in file_paths:
            # print(txt_path)
            if not '.txt' in txt_path:
                continue
            inFile = False
            for i in train_index:
                if '_%d.txt'%i in txt_path:
                    inFile = True
            if not inFile:
                continue

            emg_array = data_process.txt2array(txt_path)     # <np.ndarray> (400, 8)
            #print(emg_array)

            label = data_process.label_indicator(txt_path)
            # print(label)

            # pre-processing
            single_sample_preprocessed = data_process.preprocessing(emg_array)       # <np.ndarray> (8, 400)
            # single_sample_preprocessed = emg_array.transpose()

            # detect muscle activation region
            index_start, index_end = data_process.detect_muscle_activity(single_sample_preprocessed)
            activation_emg = single_sample_preprocessed[:, int(index_start): int(index_end)]  # (8, active_length)
            activation_length = index_end - index_start

            total_silding_size = activation_emg.shape[1] - window_size
            segments_set = []

            for index in range(total_silding_size):
                emg_segment = activation_emg[:, index: index + window_size]   # image_emg (8, window_length)
                segments_set.append(emg_segment)

            segments_set = np.array(segments_set)
            if not segments_set.any():
                continue

            if user_data is None:
                user_data = segments_set
            else:
                user_data = np.concatenate((user_data, segments_set), axis=0) # <np.ndarray>(n,8,52)

            user_labels = user_labels + [label] * segments_set.shape[0]       # <list> (n, )

        train_data_num = user_data
        train_label_num = user_labels

        # shuffle the data set
        random_vector = np.arange(len(train_label_num))
        np.random.shuffle(random_vector)
        new_data = []
        new_gesture_label = []
        new_subject_label = []
        for i in random_vector:
            new_data.append(train_data_num[i])
            new_gesture_label.append(train_label_num[i])

        # data split
        train_data = new_data
        train_gesture_labels = new_gesture_label

        # list to numpy
        train_data = np.array(train_data, dtype=np.float32)
        train_gesture_labels = np.array(train_gesture_labels, dtype=np.int64)
        # numpy to tensor
        train_data = TensorDataset(torch.from_numpy(train_data),
                                torch.from_numpy(train_gesture_labels))
        # tensor to DataLoader
        train_dataloader = DataLoader(train_data, batch_size=1024, shuffle=True, drop_last=False)

        user_data = None
        user_labels = []

        for txt_path in file_paths:
            # print(txt_path)
            if not '.txt' in txt_path:
                continue
            inFile = False
            # for i in val_index:
            if '_%d.txt'%val_index[0] in txt_path:
                inFile = True
            if not inFile:
                continue

            emg_array = data_process.txt2array(txt_path)     # <np.ndarray> (400, 8)
            #print(emg_array)

            label = data_process.label_indicator(txt_path)
            # print(label)

            # pre-processing
            single_sample_preprocessed = data_process.preprocessing(emg_array)       # <np.ndarray> (8, 400)
            # single_sample_preprocessed = emg_array.transpose()

            # detect muscle activation region
            index_start, index_end = data_process.detect_muscle_activity(single_sample_preprocessed)
            activation_emg = single_sample_preprocessed[:, int(index_start): int(index_end)]  # (8, active_length)
            activation_length = index_end - index_start

            total_silding_size = activation_emg.shape[1] - window_size
            segments_set = []

            for index in range(total_silding_size):
                emg_segment = activation_emg[:, index: index + window_size]   # image_emg (8, window_length)
                segments_set.append(emg_segment)

            segments_set = np.array(segments_set)
            if not segments_set.any():
                continue

            if user_data is None:
                user_data = segments_set
            else:
                user_data = np.concatenate((user_data, segments_set), axis=0) # <np.ndarray>(n,8,52)

            user_labels = user_labels + [label] * segments_set.shape[0]       # <list> (n, )

        val_data_num = user_data
        val_label_num = user_labels

        # shuffle the data set
        random_vector = np.arange(len(val_label_num))
        np.random.shuffle(random_vector)
        new_data = []
        new_gesture_label = []
        new_subject_label = []
        for i in random_vector:
            new_data.append(val_data_num[i])
            new_gesture_label.append(val_label_num[i])

        # data split
        val_data = new_data
        val_gesture_labels = new_gesture_label

        # list to numpy
        val_data = np.array(val_data, dtype=np.float32)
        val_gesture_labels = np.array(val_gesture_labels, dtype=np.int64)
        # numpy to tensor
        val_data = TensorDataset(torch.from_numpy(val_data),
                                torch.from_numpy(val_gesture_labels))
        # tensor to DataLoader
        val_dataloader = DataLoader(val_data, batch_size=1024, shuffle=True, drop_last=False)
        # ----------------------------------------- Training & Validation ----------------------------------------- #

        precision = 1e-8
        LR = 0.001    # learning rate

        # cnn = CNNsenet()
        if usegpu:
            model = CNNMAM().cuda()
        else:
            model = CNNMAM()

        for p in model.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=6,
                                                        verbose=True, eps=precision)

        epoch_num = 2
        patience = 12
        patience_increase = 12
        best_loss = float('inf')

        count = 0
        old_best_epoch = 1

        trains_acc=[]
        trains_loss=[]
        valids_acc=[]
        valids_loss=[]
        times=[]

        train_start_time = time.time()

        for epoch in range(epoch_num):
            epoch_start_time = time.time()
            print('epoch: {} / {}'.format(epoch + 1, epoch_num))
            print('-' * 20)

            running_loss = 0.
            correct_gesture_label, total_num = 0.0, 0.0

            # training
            model.train()
            for i, (data, gesture_label) in enumerate(train_dataloader):

                if usegpu:
                    data = data.cuda()
                    gesture_label = gesture_label.cuda()
                #data=np.transpose(data,(0,2,1))
                #print(data.shape)
                data = data.unsqueeze(1)
                pred_gesture_label = model(data)

                loss = criterion(pred_gesture_label, gesture_label)
                running_loss += loss.item()

                optimizer.zero_grad()              # clear gradients for this training step
                
                loss.backward()
                optimizer.step()

                correct_gesture_label += torch.sum(torch.argmax(pred_gesture_label, dim=1) == gesture_label).item()
                total_num += data.shape[0]

            total_acc = correct_gesture_label / total_num
            total_loss = running_loss / (i + 1)
            print('Train:   Loss: {:.4f}   Acc: {:.4f}'.format(total_loss, total_acc))

            trains_acc.append(total_acc)
            trains_loss.append(total_loss)
            # validation
            running_loss = 0.
            correct_gesture_label, total_num = 0.0, 0.0

            model.eval()
            for i, (data, gesture_label) in enumerate(val_dataloader):

                #data = data.view(1024, -1)              # torch.Size([1024, 448]) -> 每个样本变为一维向量 适配ANN的输入
                # print(data.shape)
                # time.sleep(100)
                #data=np.transpose(data,(0,2,1))
                if usegpu:
                    data = data.cuda()
                    gesture_label = gesture_label.cuda()

                data = data.unsqueeze(1)
                pred_gesture_label = model(data)
                loss = criterion(pred_gesture_label, gesture_label)
                running_loss += loss.item()

                correct_gesture_label += torch.sum(torch.argmax(pred_gesture_label, dim=1) == gesture_label).item()
                total_num += data.shape[0]

            valid_acc = correct_gesture_label / total_num
            valid_loss = running_loss / (i + 1)

            # print('Valid:   Loss: {:.4f}   Acc: {:.4f}'.format(valid_loss, valid_acc))
            # print('Time usage: {:.2f}s'.format(time.time() - epoch_start_time))
            # print()

            valids_acc.append(valid_acc)
            valids_loss.append(valid_loss)

            scheduler.step(valid_loss)
            if valid_loss + precision < best_loss:
                print('New best validation loss: {:.4f}'.format(valid_loss))
                best_loss = valid_loss
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                patience = patience_increase + epoch
                print('So Far Patience: ', patience)

            print('The best epoch is:', best_epoch)
            print('The old_best_epoch is:',old_best_epoch)
            if best_epoch ==  old_best_epoch:
                count = count + 1
                if count >= 20:
                    break
            else:
                count = 0 
                old_best_epoch=best_epoch

            print('The best epoch is equal to %d about %d times.'%(old_best_epoch, count))
            
            train_end_time = time.time()-train_start_time
        
            print('Time usage: %.2f s'%train_end_time)
            times.append(np.array(train_end_time).reshape(1,1))
            #或者保存
        # np.savetxt('times_%s.txt'%user_name,times,fmt='%.2f')
        # np.savetxt('train_loss_%s.txt'%user_name,trains_loss,fmt='%.4f')
        # np.savetxt('valid_loss_%s.txt'%user_name,valid_loss,fmt='%.4f')

        user_data = None
        user_labels = []

        for txt_path in file_paths:

            if not '.txt' in txt_path:
                continue

            inFile = False
            for i in test_index:
                if '_%d.txt'%i in txt_path:
                    inFile = True
            if inFile:
                continue

            emg_array = data_process.txt2array(txt_path)     # <np.ndarray> (400, 8)

            label = data_process.label_indicator(txt_path)
            #print(label)

            # pre-processing
            single_sample_preprocessed = data_process.preprocessing(emg_array)       # <np.ndarray> (8, 400)
            # detect muscle activation region
            index_start, index_end = data_process.detect_muscle_activity(single_sample_preprocessed)
            activation_emg = single_sample_preprocessed[:, int(index_start): int(index_end)]  # (8, active_length)
            activation_length = index_end - index_start

            total_silding_size = activation_emg.shape[1] - window_size
            segments_set = []

            for index in range(total_silding_size):
                emg_segment = activation_emg[:, index: index + window_size]   # image_emg (8, window_length)
                segments_set.append(emg_segment)

            segments_set = np.array(segments_set)
            if not segments_set.any():
                continue

            if user_data is None:
                user_data = segments_set
            else:
                user_data = np.concatenate((user_data, segments_set), axis=0)  # <np.ndarray> (n,4,8,14)
            user_labels = user_labels + [label] * segments_set.shape[0]       # <list> (n, )

        print('Loading test segments...')
        test_data_num = user_data
        test_label_num = user_labels

        # shuffle the data set
        random_vector = np.arange(len(test_label_num))
        np.random.shuffle(random_vector)
        new_data = []
        new_label = []
        for i in random_vector:
            new_data.append(test_data_num[i])
            new_label.append(test_label_num[i])

        # list to numpy
        test_data = np.array(new_data, dtype=np.float32)
        test_label = np.array(new_label, dtype=np.int64)

        #print(test_label.shape)

        # numpy to tensor
        test_data = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))

        # tensor to DataLoader
        test_dataloader = DataLoader(test_data, batch_size=1024, shuffle=True, drop_last=False)
        print('Test segments loaded. Batch_size = 1024.\n')

        print('Loading trained CNN network parameters...')


        model.load_state_dict(best_weights)
        # tcn_test = TCNw(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout).cuda()
        #tcn_test = LSTMnet(in_dim=8, hidden_dim=64, n_layer=2, n_class=6)
        # tcn_test.load_state_dict(torch.load(r'saved_model\TCNw_%s_raw_w%d.pkl'%(user_name,window_size)))
        model.eval()

        total_batch_num = len(test_dataloader)
        running_loss = 0.
        correct_gesture_label, total_num = 0.0, 0.0
        print('Start testing...\n')

        pred = []
        truth = []

        test_start_time = time.time()

        for i, (data, label)in enumerate(test_dataloader):

            #data=np.transpose(data,(0,2,1))
            gesture_label = label
            if usegpu:
                data = data.cuda()
                gesture_label = gesture_label.cuda()

            data = data.unsqueeze(1)

            pred_label = model(data)

            batch_correct = torch.sum(torch.argmax(pred_label, dim=1) == gesture_label).item()

            correct_gesture_label += batch_correct
            total_num += data.shape[0]
            # print('\tBatch Accuracy: {:.4f}\n'.format(batch_correct / data.shape[0]))
            print('\rBatch: {} / {}       Batch Accuracy: {:.4f}'.format(i + 1, total_batch_num, batch_correct / data.shape[0]), end='')
            pred_labels = torch.argmax(pred_label, dim=1)
            pred_labels = pred_labels.cpu()
            pred.append(np.array(pred_labels, dtype=np.int64))
            truth.append(np.array(label, dtype=np.int64))

        test_end_time = time.time()-test_start_time
        total_accuracy = correct_gesture_label / total_num
        print('\nTotal Segments Testing Accuracy: {:.4f},test_time:{:.4f}'.format(total_accuracy,test_end_time))

        # --------------------------------------------- Test on Samples -------------------------------------------------- #

        # print('Load Intact Samples...')

        # user_data = []
        # user_labels = []

        # # test stage
        # stride = 1
        # max_fit = 30
        # jump = 1
        # threshold = 60 / 128
        # model.eval()

        #     # subject_index = 6
        # start_time = time.time()

        # iter_times = []
        # iter_activity_times = []
        # label_prediction = []
        # label_truth = []

        # for txt_path in file_paths:
        #         # print(txt_path)
        #     if not '.txt' in txt_path:
        #         continue

        #     inFile = False
        #     for i in test_index:
        #         if '_%d.txt'%i in txt_path:
        #             inFile = True
        #     if inFile:
        #         continue

        #     emg_array = data_process.txt2array(txt_path)     # <np.ndarray> (400, 8)
        #     label = data_process.label_indicator(txt_path)

        #     sample_label = label
        #         # print('Sample   {}     True label: {}'.format(sample_index, sample_label))

        #         # if sample_label == 0:
        #         #     # relax 则跳出识别循环
        #         #     break

        #     emg_sample = emg_array        # (400, 8)
        #         # print(emg_sample)
        #         # print(emg_sample.shape)
        #     emg_preprocessed = data_process.preprocessing(emg_sample)    # (8, 400)

        #     max_sliding_length = emg_preprocessed.shape[1] - window_size  # 348
        #         # print(max_sliding_length)

        #     gesture_number_vector = [0] * number_of_classes
        #     iter = 0
        #     iter_activity = 0

        #     while iter * stride + window_size <= emg_preprocessed.shape[1]:
        #         index_start = iter * jump               # 0
        #         index_end = iter * jump + window_size   # 52
        #             # print(index_start)
        #             # print(index_end)
        #         emg_window = emg_preprocessed[:, index_start: index_end]  # (8, 52)

        #         iter = iter + 1

        #         if sum(data_process.mav(emg_window)) < threshold:
        #             pred_gesture_label = 0    # relax'
        #             pos_max = 0
        #                 # print('pass')
        #         else:
        #             iter_activity = iter_activity + 1

        #             # print(emg_spec.shape)
        #             emg_window = np.array(emg_window, dtype=np.float32)
        #             emg_spec_tensor = torch.from_numpy(emg_window,)
        #                 # emg_spec_tensor = emg_spec_tensor.cuda()
        #             #input_tensor = emg_spec_tensor.reshape(1, 416)
        #             #input_tensor=np.transpose(emg_spec_tensor) #(52,8)
        #             input_tensor=emg_spec_tensor
        #             input_tensor=input_tensor.unsqueeze(0)
        #             input_tensor=input_tensor.unsqueeze(0)
        #             #print(input_tensor.shape)
        #             if usegpu:
        #                 input_tensor = input_tensor.cuda()
                        
        #             pred_label = model(input_tensor)

        #             pred_pos = torch.argmax(pred_label, dim=1)
        #                 # print(pred_pos.item())

        #             if pred_pos.item() < number_of_classes:
        #                 gesture_number_vector[pred_pos.item()] = gesture_number_vector[int(pred_pos.item())] + 1

        #             max_num = max(gesture_number_vector)
        #             pos_max = gesture_number_vector.index(max_num)

        #             if max_num > max_fit:
        #                     break
        #             pos_max = 0
        #     iter_times.append(iter)
        #     iter_activity_times.append(iter_activity)

        #     final_prediction = pos_max
        #         # print('Final Prediction:  ', final_prediction)
        #     label_prediction.append(final_prediction)
        #     label_truth.append(sample_label)

        # count = 0
        # for i in range(len(label_prediction)):
        #     if label_prediction[i] == label_truth[i]:
        #         count += 1
        # acc = count / len(label_prediction)
        # print('Sub.{} accuracy: {:.4f}      Time Usage: {:.2f}s'.format(user_name, acc, time.time() - start_time))
                # break
    #     slices_acc.append(total_accuracy)
    #     # intach_acc.append(acc)

    # # total_acc = [slices_acc,intach_acc]
    # np.savetxt('test_mam_r_%d.txt'%loop, slices_acc, fmt='%.4f',newline='\n')