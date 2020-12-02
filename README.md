# Adversarial Frontier Stitching

This is an implemention of "[Adversarial Frontier Stitching for Remote Neural Network Watermarking](https://arxiv.org/pdf/1711.01894.pdf)"
 by Erwan Le Merrer, Patrick Perez and Gilles Trédan in TensorFlow.

### What is adversarial frontier stitching?

Adversarial frontier stitching is an algorithm to inject a watermark into a neural network. It works by first generating a set of inputs,
also called the key set which will act as our watermark.
It does that by applying a transformation, using the "fast gradient sign" method, to correctly classified inputs.
If the transformed inputs are still correctly classified we call them false adversaries and if the are now incorrectly classified we call them true adversaries.
Next we train our pretrained model on the concatenation of the training set and the true and false adversaries using their original labels
until the adversaries are correctly classified again. Our model is now watermarked. If the accuracy of the adversaries is above a predefined arbitrary threshold we verfied that the model was watermarked by us.


  

### How to use

1. Call gen_adversaries(model, l, dataset, eps) in order to generate your true and false adversary sets, which will act as your watermark.
- model is your pretrained model.
- l is the length of the generated datasets - the true and false adversary sets will both have a length of l / 2. 
- dataset is the TensorFlow dataset used for training.
- eps is the strength of the modification on the training set in order to generate the adversaries. It is used in the "fast gradient sign" method.
2. Train your model on the concatenation of training dataset and the true and false adversaries. Afterwards the model is watermarked.
3. Use verify() on the watermarked model 

### Contribute

Show your support by ⭐ the project. Pull requests are always welcome.

### License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/dunky11/adversarial-frontier-stitching/blob/master/LICENSE) file for details.
