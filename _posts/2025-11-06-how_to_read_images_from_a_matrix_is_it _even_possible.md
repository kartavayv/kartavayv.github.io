---
layout: post
title: How to read Images from a Matrix? Is it even possible?
date: 2025-11-06
categories: Questions
tags: ["Machine Learning", "Image Processing"]
---
It all started with a simple question, "Images are made up of pixels, pixels can be represented by vectors, so is it possible that a collection of vectors or a matrix can be used to read back an image — not by a computer but by me (I am human)?"

So, I tried a bunch of variations of matrices and that helped me understand how a bigger and more complex image can be represented using these simple vectors. [link of collab notebook]

I first tried making a simple grayscale image:
[code here]
Interesting thing to realize here is that each row vector of this matrix represents the a row of grayscale matrix, first row vector represents the first row and the second row vector represents the second one.  

Each of the values here can vary from 0 to 255, which is just the intensity range on a "grayscale" , 0 means completely black and 255 means completely white. But this thing becomes even interesting in RGB image:
[code of rgb image sample here]
Here things change a little bit, instead of a 2D matrix now we have a 3D matrix with 2 3*3 matrices inside it. Interesting thing here is that now each row vector represents just a single pixel in the image, all the 3 different values of the row vector are colour intensities of Red, Green and Blue colour channels. As, you may know these are called primary colours and I think you can literally make any color out of them so that's beneficial if you want to have a colorful image which can have more than just 3 colours. Also primary colours come with a benefit that now you don't need a single dimension for every single colour that exists which would have been quite wierd (I wonder if there were nothing like primary colours how would we represent different colours using vectors), you can just use 3 colours and get any colour you want, just with a vector in 3 dimensions.  

You can also turn an image into a matrix, which is quite easy using these modern libs:
[code turning image into matrix]

While I was learning this and trying to make sense of what I was looking at I came to realise that we represent images in a very different way in Numpy/Matplotlib and Deep Learning libraries like Pytorch. In Numpy like libraries the general format is ```(H,W,C)``` while in Deep Learning libs it's like ```(C,H,W)```. The simple reason for this is that Numpy or OpenCV like libraries come from the direction of image processing where we generally think in spatial grids, each pixel in space can be positioned by using it's ```(x,y)``` coordinates and then assigning an RGB value to them, as a result we intuitively think of representing images as ```(H,W,C)```. But Deep Learning libraries are optimized for fast matrix multiplication so we get a different format. [how does that format help DL libs]  [more questions emerge from this part]

So, with this another interesting question came to my mind, _**"How a resolution of an image is actually decided? Like is it somehow related to the vectors and matrices that we are dealing with here?"**_





