## Welcome to my page here I will discuss gist of some papers that I will read











### [Universal Correspondence Network](https://arxiv.org/abs/1606.03558)
Correspondence is very important task from 3d point projections between different images to semantic similarities to scene understanding, previous approaches usuualy focused on patch smilarity which usualyy takes O(n^2) passes and never takes whole image understading at once as it never sees whole image. Main key points in UCN are 1)Correspondence Contrstive loss 2) Fully convolutional NN with fast active hard negative mining 3)Fully Convolutional Patch normalization.

## Correspondence Contrastive loss:
The loss function is trying to push feature vectors corresponding to same points in two images towards each other whereas it trying to push away points atleast a distance of m who are different.

Q) Why both losses are needed? Why are we pushing vectors away from each other isn't first part of loss function i.e pulling vectors towards each other enough?
Q) Why L2 loss?
Q) How to choose margin 'M', Why SVM type loss, why not 1/d?
Q) How is the value of Si decided?
Q) Was transfer learning used or not , not clear.

## Hard negative mining
This concerns with the second part of loss fucntion, how to get pairs whose distance is less than 'M' but they are different points, Author suggests something like taking lets say 1000 points from image 1 then, computing distance to all other points in image2 and doing KNN to form blobs then take loss function value.
Q)It depends upon speed of KNN and value of 'K'?
Q) Why are we concentrating between two images first loss is taking care of that if two points correspond then they should have
same kind of vector? Other way around is in one image take all features correspoing to 'n' points then sort them according to their length? then compare only points which in window of 5 or so reducing time to O(nlog(n)).

## Experiments:
Author then performs experiments for geometric as well as Semantic Correspondence on KITTI and Pascal etc datasets. for Geometric correspondence author picks random 1000 correspondences in each KITTI or MPI images during training, If nearest neighbour in feature space is more than 16 pixels away it is considered to be negative point. for Semantic Correspondence same kind of architecture is used. In addition author also takes care of variable image sizes by proposing that a point is correctly matched if lies within a distance of alfa.L, where L is image size and alfa is metaparameter. 

## Future Works
Using this architecture get optical flow, camera pose estimation etc.

## Conclusion: with corresponding contrastive loss function author is able to reduce each feature point to corresponding embedding in preserving geometric as well as semantic similarity.



### [3D Semantic Parsing of Large-Scale Indoor Spaces](http://buildingparser.stanford.edu/method.html)

Author proposes a method for parsing the point cloud of an entire building using a hierarchical approach 
Step 1. parses the point cloud into semantically meaningful spaces (e.g., rooms, hallways, lobby) and aligns them in 3D.
Step 2. The second step parses each of these spaces into their structural and building elements (e.g., walls, columns, etc). 

## Parsing Into Disjoint Spaces:
Motivated by the peak-gap-peak signature of walls in 1D histograms of density of points projected to the major axes of a scanned area, Author form a bank of filters designed to fire in the location of walls when convolved with such signals. Author convolve each axis's histogram of density of points with filter bank which results to candidate space dividers. Author recursively merge segments uisng Connected components that belong to the same enclosed space by identifying the existence of a space divider in the connection area of two neighboring segments. This approach is highly scalable and unsupervised. 

##  Acquiring a common geometric space:
Mostly spaces within the same building have a recurrent structure and layout configuration. This structure can easily be exploited by creating a common geometric space for all rooms. Unit cube (normalization) all spaces to associate a local reference system for each semantic space (e.g. rooms, hallways etc.). 

## Parsing Spaces into Semantic Elements:
Detection rather than segmentation, 12 class(window,door etc) one vs all classifie is trained giving output for each voxel.CRF is used on top of SVM to give contexual overall informaation.

## Applications of Semantic Parsing:
Space Statistics, Estimation of Natural Illumination Model, Space Manipulation etc.



### Deep Reflectance Maps
Decomposing appearence into its intrinsic properties is a chellenging task due to difficult inverse problem. Authors here try to learn this inverse function in an end to end way using Convolutionla neural network. The input to this system is 2d image from a known object class. Author takes two approaches one is end to end using deconvolutions.For the second first get per pixel surface normal which are then used to compute sparse reflectance maps from visible normals.

## Direct approach
End to end model for predicting reflectacne map, this is a very simple end to end network similar to encoder decoder network usign deconvolutoins to get output reflectance map. The loss fucntion is L2 loss between RGB values for the predicted values and ground truth.

## Indirect approach
This consists of 4 steps:
#  1. Estimate per-pixel orientation maps from RGB image.(CNN)
 This is done end to end using CNN and taking L2 loss with ground truth. we map coordiantes to s,t coordinates of sphere.
#  2. Upsample orientation map to available image resolution.(Transformation)
 Simple upsampling was used to get to higher resolution.
# 3. Changing from image to direction domain to get sparse reflectance maps.(Transformtion)
  Here we reconstruct a sparse reflectance map from the orientation map and input image. Goal is to map samples from image to directional domain.
# 4. Predicting dense from reflectance maps from sparse maps.(CNN) SparseNet: CNN is used to convert from sparse to dense reflectance map.

Synthetic dataset were also used, This paper gives both end to end and indirect networks, Where indirect network works better because of addditional supervision which ultimately increases accuracy.



### Demon: Depth and Motion Network for learning Monocular Stereo
Taking two images to predict depth and camera motion(ego), but there is diffrence, Here first part of network predicts optical flow and warps second image based on that optical flow(disparity)(note that it is similar to case when Unsupervised dense map estimation from stereo image). Then this warped image is taken as input to net predict depth and motion. There one more part iterative part where net basically takes depth + camera(ego motion) = disparity(optical flow ) and optical flow+camera(ego motion)=depth maps and do this iteratively. 


### Learning-Based View Synthesis for Light Field Cameras

Reducing the inherent tradeoff between angular and spatial resolution. Process of estimating image at novel viewpoitn was boken into two parts disparity estimation and color estimation. These methods will allow consumer light field cameras to have low angular resolution and hence higher spatial resolution.

## Disparity estimation
First half of network is used to estimate disparity of images without ground truth. Once the disparity is estimated at novel view all input images are then warped to that novel view. All warped images should have same color at each pixels in novel view. Authors implementatin used 100 disparity levels and then warped each image to nvel view using that disparity level to generate Mean and Standard Deviation of all input warped images at each input level.

## Color Predictor
Genrating final image at novel view using disparity map and input images using CNN architecture. Input is N warpe images , disparity and novel view q.

## Training
L2 loss is used to train network using ground truth images. As there are two networks color estimtor and disparity estimator, gradients to both these networks are sent by grdient descent.

## Results
 Against a single end-to-end CNN it performs far better because normal single end to end CNN gives blurry images. Against other approaches it have egde because of task specific disparity estimator. In some cases where other methods fail this method still gives better results. Also it performs better in case of extrapolation for which it was not even designed for.


### Learning to Generate Chairs with Convolutional Networks
Using a dataset of 3D models (chairs, tables, and cars), train generative ‘up-convolutional’ neural networks that can generate
realistic 2D projections of objects from high-level descriptions.
Supervised learning and assumes high-level latent representation of the images Generate large high quality images of 128 x 128 images Complete control over which images to generate. Only Downside is the need for labels that fully describe the appearance of each image 
## Network Architecture
Targets are the RGB output image x and the segmentation mask s. Generative network g(c, v, θ) is composed of three vectors:
c: model style
v: horizontal angle and elevation of the camera position
θ: parameters of additional transformations applied to the images
2 Stream Network, from fully connected layers we are upsampling(unpooling) to RGB images and segmentaion Image. Input is high level feature description.
## Experiments
Modeling Transformations
Viewpoint Interpolation
Elevation Transfer / Extrapolation
Style Interpolation
Some Feature Space Arithmetic
Correspondences

## Analysis of the Network
To make sure that network is not just remembering the input author performs some analysis of ntwork like fixing some parameters and experimenting over some activtations in high level input. And author observes some activations were for Zoom, some for rotaiton etc.

## Conclusion
Supervised training of CNNs can be used to generate images given high-level information
Network does not simply learn to generate training samples but instead learns an implicit 3D shape and geometry representation




### [FlowNet: Learning Optical flow with convolutional networks](https://arxiv.org/pdf/1501.02565.pdf)
In this paper author proposes training CNN's end to end to learn predicting the optical flow fiel from pair of images.
The author experimented with both end to end standard CNN and Correlation CNN for optical flow. Surprisingly both networks did good job.

## Architectures
Approach is really simple given pair of iamges and ground truth learn end to end network to learn to predict ground truth. In this case ground truth is optical flow. Ony problem with this kind of network is high output space but author used refinement networks to increase the accuracy of output using gloabal(coarse) as well as local feautres.

## Contracting part:
Simple choice is to stack both input images on each other and  let network decide itself how to process it. this nertwork is called flownetsimple. Another way to create a siamese type network and later combine those feature at higher level to get output. These lower layers xtract important features from both images. Now to helparchtecture to combine those freatures in meaningfull way author proposes a corrleation layer. This correlation layer is basically with no weights and convolving one layer with another layer in small patches within a distance of d.

## Expanding part:
Upsampling is done here but also using lower layer features, So we have coarse high level features we combine them with lower layer more local feautres to increase resolution.

## Training Data:
As optical flow data is not generally available so author made some systhetic data using rotating chairs.

## Experiments:
experiments were performed on Sintel, KITTI, flying chair datasets and very highly accurate results were produced.

## Conclusion:
Author concludes by saying that network generalizes very well to non synthetic natural environments also is end to end trainanble.

Q) Logic behind strange Correlation layer why not something simple was used?

### [Predicting Depth,Surface Normals and semantic Labels with a Common Multiscale Convolutional Architecture](https://arxiv.org/abs/1411.4734)

Scene understanding is a important task. this paper solves semantic label, depth and normal prediction with shared computation in lower layers of using CNN architecture, Main Idea of this paper is using both local and coarse global featres, scale invariant loss fucntion and shared computation btw diffrent tasks.
This multiscale approach firstly takes coarse global feautres and then magnifies local features to refine them.

## Scale 1: Full Image View:
sclal1 gets get full image view because of pooling layers and fully connected layers at the end having full but coarse field of view.

## Scale 2: Predictions:
This layer only have convolution layers and no pooling or fully connected layers(there are fc layers but those are discareded after training). field of view of every feature is very small compared to features of first layer which gets full view.

## Scale 3: Higher resolutons:
This final just adds more refinement it is similar to secod layer just input is larger and magnification is higher.

## Scale invariant loss function:
as input image may have diffrent scales to protect us from that author proposed sclae invariant loss function, loss value will remain same even if you replaced d with 2d.

## Training Procedure:
Firstly author trains scale1 and 2 jointly and after fixes thes and trains scale3.

## Parameter Sharing:
Scale 1 stack is shared wiht both nornmal and depth task.

## Experiments:
Author gets state of art results over depth, normal task on NYU depth V2 datset.

## questions:
1. Why not train after scale 1 also like we did after scale 2?
2. Is there any problem where multiscale can decrease the performance as we have seen it alays incresesit?
3. Is there any other architecture which can be used in place of multisclaing like skip connections something like resnets?
4. If i replace sclale1 architure with scale2 qarchitecute and scale2 with scale1, i.e firstly get local feautres and then get global features what would be impact?


### [Viewpoints and Keypoints](https://arxiv.org/pdf/1411.6067v2.pdf)
Main Idea of this paper is combining viewpoint estimation with keypoint estimation in a way thatviewpoitn estimation provides 
global perspective about object whereas keypoint provide more of local prospect. Author is inspired by theory of of global precedence- that says humans perceive the global structure before the fine level local details. Algorithm works in two parts first author estimates the viewpoint for target object and then use this global knowledge about object with local keypoint estimation to get better keypoiny estimation.

## Viewpoint Prediction
Author formulates the problem of keypoint prediction as predicting euler angles azimuth, elevation and cyclorotation. A simple pretrained netwoork is used for this task.

## Local Appearence based Keypoint Activation
A fully connected CNN is used to model log-likelihood distribution of each keypoint locally, Only thing new here is usage of multiple scales and combining them.

## Viewpoint conditioned KeypointLikelihood
Given or after predicting viewpoint author uses gaussian mixture model to model viewpoitn conditioned keypoint likelihood.

## Keypoint Prediction
Author tries to solve both taks of keypoint Localization and Keypoint Detection using both local and global estimates computed previously.

## Viewpoint Estimation Architecture
Architecture is pretrained CNN netwotk from Imagenet with differnt fully connected layers. Outputs of this network are Nc*Na*Nq, Amazing thing is instead of training a separate CNN for each class loss layer is implemented in a way to select class corresponding angles. I haven't gone into details of this layer yet.

## Multiscale Convolutional Response maps
Author trains a fully connected CNN on multiple sclae inputs to get many convolutinal maps and then linearly combine them.

## Viewpoint Conditioned Keypoint Likelihood
Using estimated viewpoint we can say in a left facing car we can't see right wheels. Using training data of similar viewpoints we can estimate the keypoints. This data will act as prior probability. Author combines all data's which lie in geodesic distance using mixture of gaussians model.

## Keypoint Estimation
This previouslly computed data which is global act as prior probability which is then combined with more local inforamtion like keypont log likelihood to give us posterior estimation of keypoints.


## Experiments
Author performs many experiments some for keypoint and viepoint estimation with Ground truth box and without. Author gives state of art performance for both tasks.


### [Learning Dense Correspondence via 3d guided Cycle Consistency](https://arxiv.org/pdf/1604.05383v1.pdf)

Main Idea of this paper is use consistency as signal for superviosion. In this task author tackles very interesting problem of dense correspondence without much ground truth data. For each pair of training images author finds a 3d CAD model and render two synthtic views and use 4 consistency cycle to predict synthetic to real, real to real and real to synthetic correspondence using only synthetic to synthetic correspondence.
Given 3d model of two images challenge use this information to infer correspondence between two real views.

## Cycle Consistency
Cycle consistency of correspondence flows says that cmposition of flow fields for any circular path through the image set should have zero combined flow.

## Architecture
Author computes both Flow fucntion F_ab and Macthability function M_ab for this task using cycle consistency, Architecture is similar to siamese architecture. During training author applies same network to three different input pairs along cycle.(s1->r1),(r1->r2) and (r2->s2) and composite the output to optimize the consistency. While we do not know what the ground truth is we know how it should behave (cycle consistency)
Q1) Morover Idea is really great but not end to end, at training time we require 3D CAD models which can be difficult for every object.i.e we need to model every obect in model. Is there any other way in which we can force architecture to learn 3D models itself without CAD.
Q2) What about multiple objects and different objects in one image?

## Conclusion very different and great Idea to solve one task for which enough ground truth data is not available.


### [Learning to see by moving](https://arxiv.org/pdf/1505.01596v2.pdf)
Humans use visual perception for recognizing objects and perform various actions like moving around etc. Currently e have huge supervised dataset like ImageNet dataset but is it something special about labels, Is there any other supervision which can be used. biological agents perform complex visual tasks without need of supervised labels. Main Idea of this paper is Is it possible that agents can learn perceptual representations using their motor actions as supervision.
In humans and other animals brain have full access of knowledge of their egomotion(self motion) which can be taken as weak supervison for learning visual representations.
Author proposes that ueful visual representations can be learnt by using egomotion as supervion.

## Two stream architecture
Architecture is really simple it is kind of siamese architecture. Two streams both sharing same set of weights computing features from two images combined on top to classification problem of transformation. X, Y and rotation around Z axis is predicted and all three loses are taken into account. Originally a regression problem it is transformed into classification problem by using buckets.
## Q) Why classification rather than regression no reason given?

##  Datasets used KITTI and SF dataset

## Features learned uisng Egomoiton as supervison were later put on test for tasks like
# 1) Scene Recognition
# 2) Object Recogniton
# 3) Intra Class keypoint Matching
# 4) Visual Odometry
On most of the tasks this egomotion based weak superviosn outperformed Unsupervised methods

## Q) At the end author suggests something like active learning in which agent is free to move is space and decides to move to improve internal data feature representaion, Is reinforcement learning is the approach for this task, What about reward function of reinforement learning which relates to better feature representaions?
## Q) Can we use this technique for reinforcement learning pretrainin as in some reinforcemnt learning problems rewards function is very scarce, so we can use this egomotion weak supervision to learn some basic features about environement?

## Conclusion Author concludes by following results that weak supervision works for image feature learning task.


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


### [Learning to Compare Image Patches via Convolutional Neural Networks](https://arxiv.org/pdf/1504.03641.pdf)
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
# 1.Visualization of features?
# 2.Loss of information as classes information is not taken?
# 3.In 2 channel network would it ignore information present on edges which may be useful for decisio making?
# 4.No tranfer knolegde takes place.

## Conlcusion: Similarity function based on raw images only, introduced SPP which can be further extended to do multiple pyramid resolutions.




###[HD Maps: Fine grained Road Segmentation by parsing Ground and Aerial Images](http://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-5431/9230_read-46774/).
Current maps require lot of laborious man hours ti label them correctly, however an alternative is using lidar data label them correctly. But getting lidar data of all roads is very difficult task. So author proposes to use both aerial and ground images to jointly infer fine grained segmentations of roads. Author formulates problem as energy minimization problem and infer number of roads, locations of road, parking spots, sidewalks adn background as well as alignemnet between ground and aerial images from model.

## Fine grained Semantic Parsing of roads
freely available cartographic maps(osm) that gives topology of raod network.xa aerial image, xm road map,xg ground stereo images.(xm is composed of set of roads where each road is represented by piece wise linear curve representing its centerline).

## Model Formulation:
Author solves this problem using MRF(which is given below).
#  RV ={B1,s1,b2,s2,p,l1-l6,s33,b3,s4,b4} there are 15 rv's
#  graph is very simple connected graph b1-s1 s1-b2 etc
#  range of rv value each rv can take value from -15 m to 15 m from center line of OSM.
#  for ground image alignement we have t as rv which ranges between -4m to +4m.

## Energy 
energy is composed of air, ground smoothness etc.

## Aerial Semantics
the aerial semantic potential encodes tha fact that our final segmentation should agree with semantics estimated by deep net.
## Aerial edges
## Along road smoothness
## Parallel roads
## Road Collapse constraints
## Lane size constraint
## CenterLine prior


## Ground Semantics:
Author does semantics on ground image and then project images to bird eye eye and want them to align with aerial semantic images.

## BCD as optimizer.




### Markov Random fields for image analysis.
Moto: Specify locally model globally.
MRF is a countable set of RV(random variables).
## Model Ingredients:
# set of rv's
# graph of rv's
# range of each rv
# potential of each rv both single and double(although double not required) with some parameters.

## Markov blanket
Set of RV's not in clique but connected with an edge to given rv.

## Markov Property
total probability can be easily defines in terms of single potential and markov balnket double potential.

## Use EM or someother optimization to optimize given graph to find optimal values of parameters, after this we can infer everything from potentials or probabilities.



