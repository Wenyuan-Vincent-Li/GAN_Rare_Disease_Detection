# GAN Rare Disease Detection
This is a project I have done at [IQVIA](https://www.iqvia.com/), 2018 Summer. The code design follows the principle
stated [here](https://github.com/Wenyuan-Vincent-Li/Tensorflow_template).

## Model Construction
We construct a Generative Adversarial Network (GAN) for semi-supervised learning, which
could help leverage a hugh amount of un-labeled data.

We compare the results of our best model with three different classification methods, 
namely logistic regression (LR), neural network (NN(D)<sup>1</sup>), and random forest
(RF). Our proposed GAN model achieve the best result, 4.18\%, in terms of the PR-AUC
score. LR and NN(D) achieve relatively similar results, around 29\% in PR-AUC. 
While RF gives the poorest result, only 10.51\% PR-AUC is achieved. The PR curves 
are shown in the following Figure.

<img src="/Plots/Figure1.png" width="600">

We also do an extensive ablation study. The following table shows the results.
NN(D) can be understood as the 
discriminator alone in GAN model. The original GAN model is the GAN model 
without SSL branch [1], while SSL GAN stands for the semi-supervised GAN 
[2, 3]. FM is the feature matching. PT is the pull away term. Ent means the 
conditional entropy term introduced by [3].

|       Setting      | PR AUC Score |        Setting        | PR AUC Score |
|:------------------:|:------------:|:---------------------:|:------------:|
| Discriminator (NN) |    28.95%    | SSL GAN FM + PT       |    34.18%    |
| Original GAN       |    29.08%    | SSL GAN FM + Ent      |    30.20%    |
| SSL GAN FM         |    32.06%    | SSL GAN FM + PT + Ent |    30.33%    |

<sup>1</sup> _NN(D) denotes that the neural network has the same architecture as in 
our discriminator D._

#### Useful Paper：
1. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
2. [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
3. [Good Semi-supervised Learning that Requires a Bad GAN](https://arxiv.org/abs/1705.09783)

## API Serving

The Serving folder contains a flask implementation, which render a web APP that user can interact
with.

* At the home page, user can specify their own csv file location and click on Submit button to
run our model.

<img src="/Plots/Figure1.png" width="600">

* At the predict page, user will get a snippet of the csv file including two additional prediction
results (RISK_LEVEL & DISEASE_PROBS). User can download the original csv file with the prediction
padded by clicking the Download button. Click on Home button and specify another csv file to 
repeat the process.

<img src="/Plots/Figure1.png" width="600">

#### Useful Links:
