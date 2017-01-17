## Welcome to my page here I will discuss gist of some papers that I wil read



### [Do Convnets Learn Correspondence?](https://arxiv.org/pdf/1411.1091v1.pdf)
This paper main theme is do connvnets have spatial information stored in their feature space. Convnets perform very well on Classification and Object detection tasks beating all previous models, but what about visual correspondence.

It was believed that as the receptive fields of convnets are large they may loose spatial information but this paper have shown that despite having higher receptive fields convnets perform better than SIFT based correspondence approaches. In this paper author have discussed maily 4 tasks,
1. Feature Visualization
2. Intraclass alignment
3. Keypoint Classification
4. Keypoint Prediction

## Feature Visualization
In feature visualization author is trying to show that Similar Convnet features tend to have similar receptive field centers which authors show visually. Each feature vector at any layer coresponds to similar recetive field centers in original image, with increasing depth of layers and recetive field this feature space starts corresponding to more higher semantic meaning. for eg. in lower layers features have lower recetive fields and only correspond to edges etc whereas in deeper layers they start corresponding to features like nose eye etc, so if we take lets say a feature vector from 5th conv layer which whose recetive filed is centered at eye of cat, then if look for cosine similarity of this feature vector with all images in dataset and get lets say 5 nearest neigbours, from database if you see all those nearest neighbiurs receptive fields then althose recetive fields will have eyes, those eye may belong to humans or cats or dogs but sematically they will be eyes.
So we can say that feature vectors having similar meaning(cosine) in feature space, they corespong to same image patches which are present at their center of size stride\*stride.

Q) Question about this which arises in my mind is why all these feature vectors or even word vectors now even Z space by GAN follow cosine rule?

## Intraclass alignment


## Keypoint  Classification
Task of keypoint classification is to given an image and Coordinates of keypoint label that keypoint.
Author again compares this task with SIFT based methods and claims ConvNet based methods perform equally good or better than SIFT based methods. Given the coordinates of point we collect features from Conv5 layer(those features whose recetive field center lie close to those coordinates) author then train a simple SVM claasifier over those features in One VS All way and gets a Keypoitn Classifier. Also auther shows that near keypoint histogram performs far better for convnets showing better localization.

Q) Rather than selecting features based on coordinates and receptive field centers what if we train network end to end and what architecture to use for this training?

## Keypoint Prediction
Next task is keypoint prediction in this task given image we need to find keypoints in that image. Author takes each receptive filed and corresponding feature vector also we need to concatenate some more neigborig feature vectors and train SVM classifier on top of this, all receptive fields having keypoints give positive and all receptive fields not having keypoints give negative. author also discusses of increasing accuracy using Nearest Neighbour Spherical Gaussian on pool5 layer but thats not main point. Author again gets very high accuracy using this method than SIFT based method.

Q) Same type of question arises what about end to network can we use RNN for this?

## Conclusion Even with higher receptive fields convnets feture space preserve spatial information. 


### [Do Convnets Learn Correspondence?](https://arxiv.org/pdf/1504.03641.pdf)
Large datasets contain patch correspondence between images, can we make proper use of such datasets to automatically learn a similarity function for image patches?
Author aims to build patch similarity function from scratch using convolutional neural networks, with diffrent architectures.
Author explores many neural network architectures mainly two types. 1)First compute descriptor then similarity function on top of descriptor, similarity function can be L2 , SVM or fully connected layers. 2) skip part related to descriptor and directly learn proceed with similaity function.

## Model 1: Siamese:
Its very natural and easy model, theree are two branches in the network which shares  exactly the same architecture and the same set of weights. Branches can be viewed as the descriptor and top FC layers as similarity function

## Model 2:Pseudo-siamese:
This is very similar model to siamese just weights are not shared.

## MOdel3: 2-channel:
No descriptor just similarity function.

## Model 4: Central surround two stream network: 
2 streams one central one surround, central stream takes middle portion of image like magnifiation and surround streams take full image but pooled one so low resolution, this kind of architecture forces net  to focus more on centre part of image.

## Model 5: Spatial pyramid pooling:
To work with changing patch sizes author suggest image proportional image pooling, i.e if images patch size is more then pool more.

## Q) 
1.Visualization of features?
2.Loss of information as classes information is not taken?
3.In 2 channel network would it ignore information present on edges which may be useful for decisio making?
4.No tranfer knolegde takes place.

## Conlcusion: Similarity function based on raw images only, introduced SPP which can be further extended to do multiple pyramid resolutions.



