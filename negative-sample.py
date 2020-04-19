#This is used to sample negative movies that user has not interactions with, so as to balance model training process

import argparse
import math
from random import randint

def load_data (file):
    '''
    load training data
    '''
    train_dict = {}
    all_movie_list = []

    for line in file:
        lines = line.split('\t')
        user = lines[0]
        movie = lines[1].replace('\n','')
        
        if user not in train_dict:
            init_movie_list = []
            init_movie_list.append(movie)
            train_dict.update({user:init_movie_list})
        else:
            train_dict[user].append(movie)
        
        if movie not in all_movie_list:
            all_movie_list.append(movie)

    return train_dict, all_movie_list

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=''' Sample Negative Movies for Each User''')
    parser.add_argument('--train', type=str, dest='train_file', default='data/ml/training.txt')
    parser.add_argument('--negative', type=str, dest='negative_file', default='data/ml/negative.txt')
    parser.add_argument('--shrink', type=float, dest='shrink', default=0.05)

    parsed_args = parser.parse_args()

    train_file = parsed_args.train_file
    negative_file = parsed_args.negative_file
    shrink = parsed_args.shrink
    
    fr_train = open(train_file,'r')
    fw_negative = open(negative_file,'w')

    train_dict, all_movie_list = load_data(fr_train)

    fr_train.close()
    fw_negative.close()