# THIS REPO
This repo attempts to compute proximity between pairs of acido-amines using the embedding vectors of ESM.


# DATA
We have a set of ~3300 acido-amine sequences. Of these sequences we know:
 * the sequence (obviously)
 * the pairwise proximity of each element
 * the solvant accessibility (SA) of each element 


# METHODS
After inputting a batch of sequences, we find ourselves with a matrix of size (B, S, D). B is the batch size (mostly ignored for simplification), S is the sequence size, and D is the embedding size.
The goal is to estimate P: a probability of proximity.


We try different approaches to estimate the proximity between the elements of the sequence.

1. concatenate each pair of (s[i], s[j]), then go through a linear layer to estimate P.
2. pass the sequence through a pair of linear layers, resulting in vectors K and Q of shape (B, S, D'), THEN contactenating them, THEN estimating P with a cosine similarity.
3. pass the sequence through a pair of linear layers, resulting in vectors K and Q of shape (B, S, D'), THEN contactenating them, THEN estimating P with a linear layer.
4. pass the sequence through a pair of linear layers, resulting in vectors K and Q of shape (B, S, D'), THEN applying self-attention (ie: out product of K and Q resulting in a matrix of size (B, S, S)) to directly estimate P.

Using the SA, we can add an extra-step to any of the previous methods:
 1. train an MLP to estimate the SA as a 1-hot vector: 1HSA (1-hot solvant accessibility)
 2. input 1HSA + the embedding vector to the linear layers of previous methods

Indeed, it can be used as some sort of attention (not in a "feature map" sense but more in a "this result is important" way) to enrich the embedding vector.
At least for Computer Vision, it was shown that having a predictor to estimate the optical flow of a video, and then using this estimation to compute the depth of the video, improves the performance over simply asking the model to estimate the depth.

This can be intuitively understood as a way to force the network to pay attention to something that we, humans, know are important for the downstream task.
Further, the model has access to more (although "parallel") data, which is always a good thing.

# TODO
 * define part of the data as test data
 *... start training