# Adversarial Frontier Stitching

This is an implemention of "[Adversarial Frontier Stitching for Remote Neural Network Watermarking](https://arxiv.org/abs/1711.01894)"
 by Erwan Le Merrer, Patrick Perez and Gilles Trédan in TensorFlow.

### What is adversarial frontier stitching?

Adversarial frontier stitching is an algorithm to inject a watermark into a neural network. It works by first generating a set of inputs,
also called the key set which will act as our watermark.
It does that by applying a transformation, using the "[fast gradient sign](https://arxiv.org/abs/1412.6572)" method, to correctly classified inputs.
If the transformed inputs are still correctly classified we call them false adversaries and if the are now incorrectly classified we call them true adversaries.
Next we train our pretrained model on the concatenation of the training set and the true and false adversaries using their original labels
until the adversaries are correctly classified again. Our model is now watermarked. If the accuracy of the adversaries is above a predefined arbitrary threshold we verfied that the model was watermarked by us.


  

### How to use

A simple example can be found at [example.ipynb](https://github.com/dunky11/adversarial-frontier-stitching/blob/main/example.ipynb). 


1. Call [gen_adversaries(model, l, dataset, eps)](https://github.com/dunky11/adversarial-frontier-stitching/blob/1c0dd2d692ad5794d19281a6ffb6d3e9a3b2ba53/frontier_stitching.py#L15-L37) in order to generate your true and false adversary sets, which will act as your watermark, where:
* model is your pretrained model.
* l is the length of the generated datasets - the true and false adversary sets will each have a length of l / 2.
* dataset is the TensorFlow dataset used for training.
* eps is the strength of the modification on the training set in order to generate the adversaries. It is used in the "[fast gradient sign](https://github.com/dunky11/adversarial-frontier-stitching/blob/10f82d51f9433947af03a841f508c427fa82f8db/frontier_stitching.py#L5-L12)" method.
2. Train your model on the concatenation of the training dataset and the true and false adversaries. Afterwards the model is watermarked.
3. Use [verify(model, key_set, threshold)](https://github.com/dunky11/adversarial-frontier-stitching/blob/1c0dd2d692ad5794d19281a6ffb6d3e9a3b2ba53/frontier_stitching.py#L53-L66) on a model in order to test wether the model was watermarked by us, where:
* model is the model to test.
* key set is a TensorFlow dataset containing the concatenation of the true and false adversary sets.
* threshold is the p-value - it is a predefined hyperparameter in the range of zero to one which roughly controls the number of correct predictions on the key_set the model needs
in order to be watermarked by us. A lower epsilon gives more certainty to verifies prediction, but makes it also model easy for third parties to remove the watermark. Defaults to 0.05 which was used in the paper.

### Contribute

Show your support by ⭐ the project. Pull requests are always welcome.

### License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/dunky11/adversarial-frontier-stitching/blob/master/LICENSE) file for details.
