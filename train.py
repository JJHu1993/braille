import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import os
import numpy as np
from lstm_classification import LstmClassification
import matplotlib.pyplot as plt
label2text = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
              'B', 'C','P']
input_dim = 3
hidden_dim = 64
output_dim = len(label2text)
epoch = 20000
batch_size = 16
max_length = 120
def read_data():
    # plt.figure()
    train_X = []
    train_Y = []
    lengths = []
    # #load data
    files = []

    path = "./data/train/on_human/"
    files = os.listdir(path)
    for file in files:
        # if 10 < int(file.split("_")[0]) or int(file.split("_")[0]) < 3:
        #     continue
        x = np.loadtxt(path + file, delimiter=',')
        is_useful = np.sum(abs(x) > 10, 1) > 0
        start = np.argmax(is_useful)
        end = len(is_useful) - np.argmax(is_useful[::-1])
        length = end-start
        lengths.append(length)
        pad = [[0,0,0] for i in range(max_length-length)]
        x = np.concatenate((x[start:end], pad))
        train_X.append(x)
        train_Y.append(label2text.index(file.split("_")[-1].split(".")[0]))


    return np.array(train_X), np.array(train_Y), np.array(lengths)

def main():
    train_X, train_Y, lengths = read_data()
    braille_lstm = LstmClassification(input_dim, hidden_dim, output_dim, bidirectional=True).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(braille_lstm.parameters(), lr = 0.00001)
    braille_lstm.train()
    optimizer.zero_grad()
    for i in range(epoch):
        batch_index = np.array([random.randint(0,len(train_X)-1) for _ in range(batch_size)])
        batch_X = torch.from_numpy(train_X[batch_index]).float().cuda()
        batch_Y = torch.from_numpy(train_Y[batch_index]).cuda()
        batch_lengths = torch.from_numpy(lengths[batch_index]).cuda()
        batch_lengths, perm_idx = batch_lengths.sort(0, True)
        batch_X = batch_X[perm_idx]
        batch_Y = batch_Y[perm_idx]

        output = braille_lstm(batch_X, batch_lengths)

        loss = criterion(output, batch_Y)
        loss.backward()
        optimizer.step()

        #validation
        valida_lengths = torch.from_numpy(lengths).cuda()
        valida_lengths, valida_perm_idx = valida_lengths.sort(0, True)
        valida__X = torch.from_numpy(train_X).float().cuda()
        valida__Y = torch.from_numpy(train_Y).cuda()
        valida__X = valida__X[valida_perm_idx]
        valida__Y = valida__Y[valida_perm_idx]
        output = braille_lstm(valida__X, valida_lengths)
        _, result = output.max(1)
        accuracy = torch.eq(result,valida__Y).sum().item()*1.0/len(valida__X)
        print ("loss:%f  accuracy:%f%%" % (loss, accuracy*100))

        if loss < 1e-3  and i > 1000 or i == epoch - 1:
            torch.save(braille_lstm.state_dict(), "./model/braille_lstm.pt")
            break


if __name__ == "__main__":
    main()