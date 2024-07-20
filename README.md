# FsCNN-CapsNet #
A new fuzzy set convolutional neural network capsule neural network (FsCNN CapsNet) model is proposed and applied to the analysis of the sentiment analysis.

## Training ##
`/train.py`

## Evaluating ##
`./eval.py --eval_train --checkpoint_dir="./runs/1721313757/checkpoints/`

## Requirements ##
- Python 3.6.13
- Tensorflow 1.13.2
- Keras 2.2.4

## References ##
- The part of FsCNN code comes from https://github.com/jeanchen1997/FsCNN-2-classification
- The part of CapsNet code comes from https://github.com/khikmatullaev/CapsNet-Keras-Text-Classification
- The following URL we quote for datasets   
  https://github.com/khikmatullaev/CapsNet-Keras-Text-Classification/tree/master/datasets   
  https://github.com/AcademiaSinicaNLPLab/sentiment_dataset/tree/master/data    
  https://www.kaggle.com/datasets/williamhua/senteval     
 
