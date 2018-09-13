# GAN Rare Disease Detection
This is a project I have done at IQVIA, 2018 Summer. The code design follows the principle
stated [here](https://github.com/Wenyuan-Vincent-Li/Tensorflow_template).

We construct a Generative Adversarial Network (GAN) for semi-supervised learning, which
could help leverage a hugh amount of un-labeled data.

We compare the results of our best model with three different classification methods, 
namely logistic regression (LR), neural network (NN(D)<sup>1</sup>), and random forest
 (RF). Our proposed GAN model achieve the best result, 4.18\%, in terms of the PR-AUC
  score. LR and NN(D) achieve relatively similar results, around 29\% in PR-AUC. 
  While RF gives the poorest result, only 10.51\% PR-AUC is achieved. The PR curves 
  are shown in the following figure.

![Model Comparison](/Plots/Figure1.png =250x)
<img src="/Plots/Figure1.png" width="200">

\footnote{NN(D) denotes that the neural network has the same architecture as in 
our discriminator D.} 


#### Useful Paper：
[]
[]