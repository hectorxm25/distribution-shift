Fill in later

note for supercloud usage:
using the anaconda module anaconda/Python-ML-2024b
using the distrib-shift venv

NOTE: Pre-trained RESNET18 models are trained on ImageNet, which are much larger than CIFAR10 images. We must preprocess them and probably retrain a RESNET18 model for it to work. 
TODO: train a resnet18 model on the locally-downloaded CIFAR10 dataset then generate adverserial examples on that model, then train a model on a 50/50 split of adversarial/natural datapoints.
However, first I will use a pretrained resnet18 model on CIFAR10

NOTE: CIFAR10 dataset is not entirely in this repo, look at the adversarial/temp_example.py way to download the dataset locally.

This (link)[https://www.perplexity.ai/search/i-want-to-get-adversarial-exam-yeB0IQfgRYC5eka1SLhT8g] has a lot of what was used to code this

More notes (to be stored elsewhere):
Accuracy on test set for RESNET18 in-house trained model is 95%