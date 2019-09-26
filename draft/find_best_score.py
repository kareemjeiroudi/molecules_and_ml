#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:39:55 2019

@author: kareem
@Date: 04.03.2019
@Title: Best Score out of the Grid Search
"""

def collect_score(file_path):
    scores=[]
    f = open(file_path, 'r')
    for line in f.readlines():
        if line.startswith('auc'):
            scores.append(float(line.split(' ')[2]))
    f.close()
    return scores

def getBestScore(scores):
    best_score = 0.0
    for score in scores:
        if score > best_score:
            best_score = score
    return best_score

scores = collect_score('OUTPUT_02.txt')
print(len(scores))
print(getBestScore(scores))