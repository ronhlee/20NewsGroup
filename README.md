# 20NewsGroup Text Classification
Text classification of news documents from 20 News groups. Some of the amazing (IMHO) concepts/techniques I learned in fastai deep learning course such as Stochastic Gradient Descent with Restarts will be utilized.

## Acknowledgments
As always I would like to thank the following entities for giving me such an easy chance to learn ML/DL/AI. I would not be able to do this without them.
* Andrew Ng and the team @ Deeplearning.ai
* Jeremy Howard, Fastai team, and its community
* ML/DL/AI researchers, teachers, and contributors among the world wide web

## Getting Started

### Library

Things you need to install to run this notebook: [fastai version 1.0](https://github.com/fastai/fastai). Follow instruction from github page

### Dataset

20NewsGroup originally from Ken Lang's [collection](http://qwone.com/~jason/20Newsgroups/) contains news documents from 20 different news group. 

Import 20NewsGroup dataset from Scikit-Learn right from the notebook:
```python
from sklearn.datasets import fetch_20newsgroups
```
## Walk Through
### Pre-processing
After defining label and data in the dataset
```python 
df = pd.DataFrame({'label':dataset.target, 'text':dataset.data})
```
and trimming it to be the binary classification problem (taking two labels - ***1.*** ‘comp.graphics’ and ***10.*** ‘rec.sport.hockey’ - out of twenty labels) by
```python
df = df[df['label'].isin([1,10])]
```

Then we split the dataset into training and test set:
```python
from sklearn.model_selection import train_test_split
# split data into training and validation set
df_trn, df_val = train_test_split(df, stratify = df['label'], test_size = 0.4, random_state = 12)
```
Fastai library provides amazing pre-processing class `TextDataBunch` that takes care of both tokenization and numericalization.

Tokenization is to keep track of words (including words, punctuation, and what not) that are contained in our raw text data as tokens --- for which Fastai utilizes powerful industry-grade [Spacy tokenizer](https://spacy.io/api/tokenizer).
Numericalization just maps a unique id to each token and vice versa.

### Training
To train a neural net to understand English and be able to tell if the text describing computer graphics or hockey sport, we're going to so much more than this small amount of data we have (~600 docs for each class). So we take a shortcut ... Transfer Learning. We use a pre-trained language model (WT103), that's trained using Wikipedia text and at least understand English text, to train for an epoch with only the last layer unfreeze so that we get some meaningful weights in our last layer:
```python
learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.7)
learn.fit_one_cycle(1, 4e-2)
``` 
Note `4e-2` is the learning rate (lr) which is one of the most important hyper-parameters to set in training neural netse. Set it too low, you'll burn too much time/cash(if you train on a cloud service) optimizing your algorithm. Set it too high, you will never find your algorithm optimized!! That's where Fastai's `lr_find()` function comes to rescue. It basically runs the algorithm for a range of lr with a small amount of iteration per lr to returns the loss vs learning rate plot like the one shown in the notebook. And from that plot, we want to choose the high enough lr for which the loss is still decreasing which is where `4e-2` comes from. We save the encoder from this trained language model to use in our classification task. 

Then, create a classifier model data `data_clas` using `TextClasDataBunch` and pass it through `text_classifier_learner` setting the encoder to the one we got from training language model:
```python
data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=bs)
learn = text_classifier_learner(data_clas, drop_mult=0.7)
learn.load_encoder('ft_enc')
```
To train our classifier, we'll be using a few powerful techniques taught in Fastai course [1]. First, you can see that the training is done one epoch at a time with each time lr decreasing and more layers are unfrozen starting from the last layer. The point is to train using differential lr -- the latter layers with higher lr and the earlier layers with lower lr -- leveraging the idea that earlier layers of neural networks tend to learn basic features in the data (thus need less training from the transferred language model baseline) while the feature learned tend to get sophisticated in latter layers (thus need more training). You can explore more of this idea in the paper by Matthew Zeiler *et al* [3]. 

Second, the lr input, instead of one constant value, is a slice object that defines the min and max lr bound for cyclic lr schedule. The paper SGDR: Stochastic Gradient Descent with Restarts by Loshchilov *et al*[2] explores the effects of cyclic lr schedule improving the training time and accuracy. So how do gradually decreasing and cyclically restarting the lr help? The paper and Fastai lectures extensively cover this but , they help because an increase in the learning rate cyclically ensure that we escape
- the saddle points -- the points where the derivatives of all axes of the loss function become zero but they are not the local minima on all axes and 
- bad local minima -- the sharp local minima which do not generalize well to new data

Finally, the following code does what described above
```python
learn.fit_one_cycle(1, 1e-2)

learn.freeze_to(-2)  # unfreeze up to the second to the last layer
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))

learn.freeze_to(-3)  # unfreeze up to the second to the last layer
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))

learn.unfreeze()  # unfreeze all layers
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))
```
At the beginning of the training in the notebook, I put a markdown that shows the results of training using a constant learning rate for 12 epochs straight. You can see that that result is about the same as what we now get by using the techniques described above in only 5 epochs.  

Also note that our training loss is still pretty high compared to our validation loss. I ran out of time/money for my Tesla P100 :( but you can still continue to train until training loss is close to validation loss. Happy training!!

## Reference
[1] [Fastai Course](https://course.fast.ai) & [Fastai Docs](https://docs.fast.ai)

[2] I. Loshchilov and F. Hutter. Sgdr: Stochastic gradient descent with restarts.  
_arXiv preprint arXiv:1608.03983, 2016_.

[3] M. Zeiler and R. Fergus. Visualizing and Understanding Convolutional Networks https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
