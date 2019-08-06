# AutoAI
This repository is a compilation of scripts that I have created in my time working with machine learning. These scripts aim to automate the annoying and silly parts of ML, allowing you to focus on what is important.
<h2> AutoAi.manual_test(model, testing_dir, labels) </h2>
<h5> This function tests a model given labels and testing data. It then compiles the results in a CSV file, and groups the results by class, and by correct and incorrect.</h5>
<ul> 
  <li> Model - Path of model that you want to test or model object.</li>
  <li> Testing_dir - Path to the directory with your testing data.</li>
  <li> Labels - Dictionary of the classes, in form (index:class_name)</li>
  </ul>
