# squeezenet v1.1 trained in places365
This repository contains weights for a variant of [squeezenetv1.1](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1) trained on the [places365](https://github.com/metalbubble/places365) dataset.

I made some changes to the original squeeznet network definition:

    - BatchNormLayers after each ConvLayer
    - rrelu instead of relu

The training is not yet complete, but the performance is already quite good to use the weights for transfer learning tasks. 
I probably will post updated weights in the future.

Running the evaluation [script](https://github.com/metalbubble/places_devkit/blob/master/evaluation/demo_eval_cls.m) I get the following errors Top1-Top5 errors:

Top1 accuracy: ~40%
Top5 accuracy: ~71%

# guesses vs cls error
    1.0000    0.6051
    2.0000    0.4646
    3.0000    0.3841
    4.0000    0.3297
    5.0000    0.2904

# guesses vs accuracy
    1.0000    0.3949
    2.0000    0.5354
    3.0000    0.6159
    4.0000    0.6703
    5.0000    0.7096

For comparison ResNet152 accuracy: 
Top1 ~55% 
Top5 ~85%
