## Intro

[Accompanying course book to this course](https://www.learnpytorch.io/)

[Repo with code, exercises and slides](https://github.com/mrdbourke/pytorch-deep-learning) and [discussion](https://github.com/mrdbourke/pytorch-deep-learning/discussions)

[google colab](https://colab.research.google.com/)

### What will this course cover?

- Pytorch basics and fundamentals / tensors and tensor operations
- preprocessing data
- building and using pretrained deel learning models
- fitting a model to data (learning patterns)
- making predictions with a model (using patterns)
- evaluation of model predictions
- saving and loading models
- using a trained model on custom data

### General terms

- Machine learning is turning data into numbers and finding patterns in those numbers

**Artificial Intelligence** is a _superset_ of **machine learning** which _is a superset_ of **deep learning**

- Traditional programming: inputs -> rules -> output
- Machine learning (supervised learning): Inputs + known outputs -> rules (that can be applied to more inputs with unknown outputs)

Outputs: also known as features / labels

### Why use machine learning?

- Some problems are too complex / numerous or the rules are simply not known to build them algorithmically
- continually changing environments / rules
- large datasets (same as "too many rules")
- discovering insights in large datasets (same as "unknown rules")

### When not to use ML

- If a simple rule-based system can be built w/o ML, do that (google ML rule from [handbook](https://developers.google.com/machine-learning/guides/rules-of-ml))
- If explainability is required, e.g. financial transactions. Usually a trained ML model is an uninterpretable  blackbox.
- If errors are unacceptable (outputs are probabilistic, so they _can_ be wrong occasionally)
- When not much data with known outputs is available for the given problem

### Machine learning vs deep learning

#### Machine learning

_Typically..._

- uses structured data (e.g. a spreadsheet)
- typical algos ("shallow algorithms"): 
  - XGBoost (gradient boosted machine)
  - random forest
  - naive bayes
  - nearest neighbor
  - support vector machine
  - ...

#### Deep learning

_Typically..._

- Unstructured data (such as texts, images, audio), though images / audio are or can be turned into structured data
- uses neural networks such as
  - fully connected NNs
  - Convolutional NNs
  - Recurrent NNs
  - Transformer
  - ...

#### Overlap

Depending on how a problem is represented, many algorithms can be used for both structured and unstructured data, e.g. a text could be converted into a table with word counts and positions, and an image is basically a bitmap (x and y positions and color information in the "cells") anyway.

### What are neural networks?

- Unstructured data (e.g. Texts) (=**inputs**) is turned into numbers (matrices / tensors)
- neural network (input layer -> hidden layers (optional) -> output layer): learns representation (patterns / features / weights)
- creates representation outputs (features, weight matrix)
- outputs can be converted in human understandable terms, such as "image is of a bird"

![what are nns](./readme_images/what_are_nns.png)

### Anatomy of neural networks

- The "deep" in deep learning comes from the typically huge amount of hidden layers

![nn anatomy](./readme_images/nn_anatomy.png)

### Types of learning

- Supervised learning
  - typically if there are lots of existing samples with known labels / outcomes
- un-supervised and self-supervised learning
  - typically just the data without labels is present. Learns patterns (e.g. clusters) only from the data itself. Labels are later assigned by humans to the patterns.
- Transfer learning
  - Takes patterns of an existing model and tries to apply them to a different problem
- Reinforcement learning
  - >Reinforcement learning is a machine learning training method based on rewarding desired behaviors and/or punishing undesired ones. In general, a reinforcement learning agent is able to perceive and interpret its environment, take actions and learn through trial and error.

The course will focus on supervised learning and transfer learning. [source](https://www.techtarget.com/searchenterpriseai/definition/reinforcement-learning)

### What is deep learning used for?

- recommendation
- translation
- language models (chatgpt)
- seq2seq = sequence to sequence = sequential data (audio) in, sequential data (text) out
  - speech recognition 
- classification / regression
  - computer vision (self driving cars etc)
  - natural language processing (NLP), e.g. for spam detection

### Pytorch

[pytorch home](https://pytorch.org/)

- most popular research deep learning framework
- fast (can run on GPUs using cuda and TPUs = tensor processing unit) 
- can access pre-built models from [torch hub](https://pytorch.org/hub/)
- full stack from preprocessing data to model deployment

[good general ML resource: paperswithcode](https://paperswithcode.com/sota)


### What is a tensor?

- almost any representation of data

![tensor](./readme_images/tenso_1r.png)

[what is a tensor video](https://www.youtube.com/watch?v=f5liqUk0ZTw)

### Pytorch workflow

![pytorch workflow](./readme_images/pytorch_workflow.png)

## Google colab

- Use GPUs: Runtime -> change runtime type
- convert cell to text cell: **ctrl-m m**
- run cell: ctrl-enter
 
[00_pytorch_fundamentals_video.ipynb](https://colab.research.google.com/drive/16H1vO_v4m_2oHxCcu6jl-G4XtrJZvuSp)

