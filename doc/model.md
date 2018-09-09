## Inputs to classification model

3D tensor representing a history of recent notes played/playing:

0. _batch_
1. time
2. channel
3. sample data; vector of
  - 1-hot encoding of note data
  - instrument/voice
  - strength of attack


## Output from classification model

1D tensor representing current musical style:

0. _batch_
1. Parameters: support vector; embedding of style

> Q: Should this have a time dimension, rather than just capturing the "current" or overall style? Should this also consider the future (which is known during training)?

## Inputs to generative model

3D tensor representing a history of recent notes played/playing:

0. _batch_
1. time
2. channel
3. sample data; vector of
  - 1-hot encoding of note data
  - instrument/voice
  - strength of attack

1D tensor representing current musical style:

0. _batch_
1. Parameters; support vector; embedding of style


## Output from generative model

2D tensor representing the next notes to play:

0. _batch_
1. channel
2. sample data; vector of
  - 1-hot encoding of note data
  - instrument/voice
  - strength of attack

> Q: Should the generative model also output the current style, to ensure that the style that was input is being used
