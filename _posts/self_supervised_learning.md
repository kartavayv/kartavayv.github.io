---
layout: post
title: Self Supervised Learning
date: 2025-09-14
category: Machine Learning
tags: ["Self Supervised Learning", "Project"]

---
{interesting img here}

This week I have embarked on the journey of building something different. I was thinking of building something hard, I have built a [neural net](https://kartavayv.github.io/machine%20learning/2025/08/30/What-I-learnt-by-Building-a-Neural-Net-from-Scratch.html) from scratch before, but I didn't feel that push which I want to.  

This week I have started building a self supervised learning model, which is quite different than a simple neural net and a little bit harder also.  

# What is Self Supervised Learning?

To train a ML models we need a lot of data, most of this "lot of" data is labeled data which is used to train the ML model and serves as a feedback for our model. But there is a lot of unlabeled data being generated every second on the internet, and not using it would be a mistake. So we need some kind of method to harness the power of this vast amount of un-labeled data on the internet and get the desired results from it. This is where self supervised learning comes in.  

Self Supervised Learning helps us to train the models which can learn from the unlabeled data. You can train a model using SSL on some broad task and then train it on some downstream task (a specific task), this way you won't need to gather labeled data for that specialized task, also since the tasks are of the same type the model which is trained on the broader task has already learnt the important patterns in that kind of data, now you just need to train it on the specific task using labeled data, which is easy and does not require that much amount of labeled data.  

The process of training a model on broader tasks is called pre-training and then focusing it down on a smaller, narrow task is called finetuning. {really...?}  

# How a SSL model learns?

The general pattern of learning is same in SSL as is for other supervised or unsupervised learning tasks. You put the inputs in the model, make predictions, calculate error, do backpropagation, and repeat it as many times as you want. But in SSL we don't have labels to our features, because we are working with unlabeled data, so it becomes quite difficult to calculate errors on your predictions. In case of labeled data we had the labels for the features to compare accuracy of our predictions against, but now we have to figure out some other method to do it.  

The solve this problem we will need to just change the structure of our cost function.  

I have described the structure of the project below.

# Building an SSL model

I am building this whole project in sort of OOP way, so our model will be a class named SSL and inside it we will have everything that we are going to use in this project. I am using PyTorch to build this model.  

## Creating Batches

Our model will take the input from the user, convert it to tensors and give us it in batches. Following is the function we will use for that.  
```python

  def create_views(self):
    data = iter(self.train_loader)

    tr1 = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(degrees=180), transforms.functional.adjust_brightness(123)])
    tr2 = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(501)])

    aug_data_batch = []

    for img_batch, batch_labels in data:

        b1,b2 = [tr1(img) for img in img_batch], [tr2(img) for img in img_batch]

        [aug_data_batch.extend([i,j]) for (i,j) in zip(b1,b2)]

    self.tf_img_batch = tuple(aug_data_batch)

```
Since this function is a part of a larger class called ```SSL```, so here is the structure of that __init__ function, you will soon know the purpose of all the other elements of this class.  
```python
class SSL():

  def __init__(self, input_data, batch_size=64,project_dim=128):
      self.data = input_data
      self.train_loader = DataLoader(input_data, batch_size=batch_size, shuffle=True)
      self.tf_img_batch = ()
      self.encoder = self.Encoder(self).forward()
      self.embeddings = []
      self.project_dim = project_dim
      self.projection_head = self.ProjectionHead(self)
      self.projected_output = [] #is it okay to create so many attributes related to just one functionality
      self.loss_calc = self.Loss(outer=self)

```
As you can look in the first code snippet that I am saving the outputs of ```create_views``` in ```tf_img_batch``` which is a tuple, I am not sure what would be a better way so I just chose a tuple. Also you can see that in ```create_views``` I am using transformations on the elements of each batch and the reason to do that is because this will help us to learn from unlabeled data. 


{explain here why you said what's above }
