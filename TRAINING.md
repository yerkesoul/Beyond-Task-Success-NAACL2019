# Training
Here we will walk through the process of training your own model. Only the process for training your Supervised Learning (SL) model and inference with that model is being described here.

## Preprocessing
The image features for the train/val/test split have already been precomputed and can be downloaded from here, https://aashishv.stackstorage.com/s/3HHpnNTjVGleSAU . Place these files in the data folder.


You should also be able to extract the image features using the file **utils/ExtractImgfeatures.py**. This however, has not been adjusted to the different dataset and does not work at the moment.

## Training
You can only train your SL model. Documentation for learning a CL model is missing.

### Supervised Learning (SL)
The SL model can be trained using two methods, modulo-epoch and modulo-iteration.

To train model with modulo-epoch use the command:

`python3 -m train.SL.train -modulo 7 -no_decider -exp_name xxx -bin_name xxx`


Here, `-exp_name` is used to store all the tensorboard logging with this name and -bin_name saves the models with this name.

For testing the code on your system flags like `-my_cpu` and `-breaking` might be helpful as the use only a subset of all training points to check the entire flow of the training script. Also in absence of precomputed image features you can use the `-resnet` flag to compute the features on the fly.


### GamePlay(Inference)
GamePlay is interaction between the Questioner and Oracle to figure out the target object. To do this interplay the code in the GamePlay folder is used. 

To run the Game use the command:

`python3 -m train.GamePlay.inference -exp_name xxx`

Again for debugging you can use the `-breaking` flag and if you are running this on your personal computer be sure to also set `-my_cpu`. This will give a log of the generated games for validation and test sets. If you want to log the encoder hidden state at each question in every game you can do it by using the flag `-log_enchidden`.

If you have multi-GPU node then you can use `-dataparallel` flag to run the inference with bigger batch size. 
Also, if you are trying to check the GamePlay score for various models then you can use the script in **utils/run_inferenceN2N.py**.
