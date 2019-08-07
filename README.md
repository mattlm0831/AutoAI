# AutoAI
This repository is a compilation of scripts that I have created in my time working with machine learning. These scripts aim to automate the annoying and tedious parts of ML, allowing you to focus on what is important.
PyPi: https://pypi.org/project/AutoAILib/
</br> $ pip install autoailib </br>
This library was developed for and used with keras convolutional neural networks. They do however work with other keras models, besides image test obviously.
<h2> AutoAi.manual_test(model, testing_dir, labels) </h2>
<h5> This function tests a model given labels and testing data. It then compiles the results in a CSV file, and groups the results by class, and by correct and incorrect.</h5>
<ul> 
  <li> Model - Path of model that you want to test or model object.</li>
  <li> Testing_dir - Path to the directory with your testing data.</li>
  <li> Labels - Dictionary of the classes, in form (index:class_name)</li>
  </ul>
  <h5>Example csv:</h5>
  <img src="https://i.imgur.com/g4gNQjS.png"></img>
<h2>Update! This has now been packaged in the AutoAI.data_compiler class.
  AutoAi.data_compiler(self,src, dest, **kwargs)</br>
  AutoAi.data_compiler.run() will compile the data based on the constructor parameters. </h2>
<h5> This function takes 2 required arguments, an original data source file, and a path to the desired data directory. Given just these two arguments, this function will create a new testing data folder at dest with training, validation, and testing folders, containing folders for each class. You can alter the ratio with the ratio arguments, as well as provide a number of img transforms to do if you are using images.</h5>
<ul>
  <li> Src - Path to a folder that contains a folder for each class and then data examples in those class folders. </li>
  <li> Dest - Path to a folder where you want the data to end up. </li>
  <li> Num_imgs_per_class - This number of images will be added to the original set for each class through transforms. The theoretical limit for this would be 3! * original images per class </li>
  </ul>
  <h5> Demo:</h5>
  Given the a path to the following folder:
  <img src="https://i.imgur.com/SSpydEv.png"></img>
  If augmentation used the following results will be yielded:
  <img src="https://i.imgur.com/4okyMrN.png"></img>
  Then these images will be copied to the dest folder with copied file structure, but an added upper layer:
  <img src="https://i.imgur.com/TY7HvL4.png"</img>
  Example showing the images made it:
  <img src="https://i.imgur.com/3ily5dU.png"</img>
  
  
  <h2> AutoAi.image_predict(model_path, image_path, labels)</h2>
  <h5> This function takes 3 arguments: a path to a keras model, a path to an image, and a list of labels.</h5>
  <h5> Demo:</h5>
  Given a the correct arguments, we get the following output, as well as this image saved to our model directory.
  <img src="https://i.imgur.com/woiPdus.png"></img>
