# squeezenet v1.1 trained in places365
This repository contains weights for a variant of the [squeezenetv1.1](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1) trained on the [places365](https://github.com/metalbubble/places365) dataset.

I made some changes to the original squeeznet:

    - BatchNormLayers after each ConvLayer
    - rrelu instead of relu

The training is not yet complete, but the performance is already quite good to use the weights for transfer learning tasks. I probably will post updated weights in the future