# sequences
A notebook and code for exploring RNN sequence predictors

![alt tag](/resources/in_train.png)
 
I’ve come to machine learning sequence models though image labeling in particular the excellent [Andrej Karpathy CS231n](https://www.youtube.com/watch?v=ByjaPdWXKJ4) lectures, but it’s pretty obvious the first time you see them how useful they should be in a whole range of problem spaces. 

Given a sequence of previous events predict future events.

Pretty general I think you’ll agree.

To play around with some ideas that popped into my head I grabbed the Air Quality data set from [http://archive.ics.uci.edu/ml/datasets.html](http://archive.ics.uci.edu/ml/datasets.html) 

![alt tag](/resources/air_quality.png)  

It didn’t really matter what the data was about, only that it was fairly noisy time series data.

To start I picked a single dimension and trained an lstm to recognise what will come next.

# What choices are available to us?

How big an input sequence? Well, there’s a decent chunk of data to work with so we’re not really constrained.
How far into the future can we look? I wanted a decent feel of prediction and not just a what comes next result.

If we chose 20 and 7 it would look like:
						
![alt tag](/resources/labels.jpg)

and we are predicting all the values 21-27, at the next time step our inputs are 2-21 and we predict 22-28

A mean squared error seems reasonable for this data.

You quickly run up against all the usual machine learning techniques - a pretty small lstm network quickly learns the training data well so we’ll need regularization and dropout. We also need to batch our data: without it we can get pretty good initial learning but there is a high risk of instability later on. 

![alt tag](/resources/losses.png)

Results of increasing batch size from 4 to 8 to 12. Significant reduction in noise while learning and better convergence.

# LSTMs GRUs Normalization:
![alt tag](/resources/squiggels.png)

Some initial guessing: the real values are in Yellow, Red looking furthest into the future.
Even without regularization or dropout, the BasicLSTM is starting to do some good work.

It’s nice and easy in tensorflow to swap different rnn cells in and out - I found GRUs *much* faster to train than ordinary LSTMs and normalized LSTMs really slow.

# So what else can we do?

Let’s assume the gases are related in some way: given data from a few of the gases present can we predict the future levels of a different gas?

To do this we create dynamic rnns for each input gas, combine them with a sum and learnable multipliers for each gas's effect and adjust our mean squared error using label data from the target gas. 
The multiple inputs to our final logits increase the complexity slightly but tensorflow effectively removes the effort of thinking about the gradient flows (a good and a bad thing).

It works well but if we look at the normalised input data this isn’t too surprising, there are a lot of similarities in the data, but it’s nice that it works.

These are out of training predictions: the blue trace is the real values, red our predictions: remember at any stage we are looking 26 time steps forward so for instance at timestep 51 we are predicting the peak at around 77. 

![alt tag](/resources/combined.png)

I swapped in and out a few different gas mixes - the logit multipliers nicely follow visual intuition of how useful one gas will be in predicting the actions of another.

I ran the [notebook](/notebooks/sequence_notebook.ipynb) in a standard tensorflow docker image: [https://hub.docker.com/r/tensorflow/tensorflow/](https://hub.docker.com/r/tensorflow/tensorflow/)

# What are we learning for?

How much regularization we apply can depend on what we want the data to do. If we hope the time series continues much as before the purpose may be to spot anomalies and less regularization will help highlight change.

The low regularization example below tracks much of the target set very closely but areas of large difference increase loss significantly - which may be what we want.

![alt tag](/resources/short_term.png)

