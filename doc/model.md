## Inputs to classification model

3D tensor representing a history of recent notes played/playing:

0. _batch_
1. time
2. sample data
 - voice
 - probability (always 1)
 - velocities of pitches from 0 (off) to 1 (fastest)
3. track (analogous to channel in image processing)


## Output from classification model

1D tensor representing current musical style:

0. _batch_
1. Parameters: support vector; embedding of style

> Q: Should this have a time dimension, rather than just capturing the "current" or overall style? Should this also consider the future (which is known during training)?

## Inputs to generative model

3D tensor representing a history of recent notes played/playing:

0. _batch_
1. time
2. sample data
 - voice
 - probability (always 1)
 - velocities of pitches from 0 (off) to 1 (fastest)
3. track (analogous to channel in image processing)

1D tensor representing current musical style:

0. _batch_
1. Parameters; support vector; embedding of style


## Output from generative model

2D tensor representing the next notes to play:

0. _batch_
1. sample data, concatenation of:
 - voice
 - probability that this is the appropriate sample to play next
 - velocities of pitches from 0 (off) to 1 (fastest)
2. track (analogous to channel in image processing)

> Q: Should the generative model also output the current style, to ensure that the style that was input is being used
