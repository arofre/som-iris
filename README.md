# Self-Organizing Map (SOM) on Iris Dataset

Implemented a Self-Organizing Map from scratch to cluster Iris flower data in an unsupervised setting.

## Overview
- Input: 4D Iris features  
- Output: 40×40 neuron grid  
- Training: Gaussian neighborhood updates with decaying learning rate and radius  

## Method
- Find Best Matching Unit via Euclidean distance  
- Update BMU and neighbors  
- Train over multiple epochs with stochastic updates  

## Result
The model learns a 2D representation where samples form distinct clusters corresponding to the three Iris species.
