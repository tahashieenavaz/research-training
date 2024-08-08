# Wisdom Of Ensemble: Improving Consistency Of Deep Learning Models

Deep learning models are assisting humans in their decision making process, therefore user trust is of paramount importance. User trust is sometimes a function of constant behavior.

As this sounds like a straightforward requirement as we retrain models over time there are no guarantees that they produce the same **correct** output for the same input. In this paper consistency has been defined as the ability of a model to generate correct samples over the course of its maturity during different cycles of training. By way of an illustration, imagine that a model classifies a picture correctly and we will run an update over night, say, for car drivers, and in our latest version the model is NOT able to classify something that it could detect correctly yesterday. This would greatly affect drivers' trust.
