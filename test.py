import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from lstm_classification import LstmClassification

label2text = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
              'B', 'C','P']
input_dim = 3
hidden_dim = 64
output_dim = len(label2text)




def read_data():
    test_X = []
    test_Y = []

    path = "./data/test/on_human/"
    # path = "./data/train/set5/"
    files = os.listdir(path)
    for file in files:
        # if int(file.split("_")[0]) == 16:
        test_X.append(np.loadtxt(path+file, delimiter=','))
        test_Y.append(label2text.index(file.split("_")[-1].split(".")[0]))

    return np.array(test_X), np.array(test_Y)

def main():
    test_X, test_Y = read_data()
    braille_lstm = LstmClassification(input_dim, hidden_dim, output_dim, bidirectional=True).cuda()
    braille_lstm.eval()
    braille_lstm.load_state_dict(torch.load("./model/braille_lstm.pt"))
    results=[]
    plt.figure(figsize=(10,5))
    for test_x, test_y in zip(test_X, test_Y):
        # is_useful = np.sum(abs(test_x) > 10, 1) > 0
        # start = np.argmax(is_useful)
        # end = len(is_useful) - np.argmax(is_useful[::-1])
        # length = np.array([end - start])
        # output = braille_lstm(torch.from_numpy(test_x[start:end]).unsqueeze(0).float().cuda(), torch.from_numpy(length).cuda())
        length = np.array([len(test_x)])
        output = braille_lstm(torch.from_numpy(test_x).unsqueeze(0).float().cuda(), torch.from_numpy(length).cuda())
        _, result = output.max(1)
        result = result[0].item()

        print("predict: %s, groundtruth: %s" % (label2text[result], label2text[test_y]))
        #plot
        # sf_output = F.softmax(output[0], dim=0)
        # plt.bar(label2text,sf_output.cpu().detach().numpy())
        # plt.bar(label2text[result],sf_output[result].cpu().detach().numpy(), color='r')
        #
        # plt.draw()
        # plt.pause(0.1)
        # plt.clf()

        # print("predict:%d   groundtruth:%d" % (result, test_y))
        # plt.plot(test_x)
        # plt.show()
        results.append(result)


    accuracy = np.sum(np.equal(results, test_Y)) * 1.0 / len(test_X)
    print("accuracy:%f%%" % (accuracy * 100))





if __name__ == "__main__":
    main()
