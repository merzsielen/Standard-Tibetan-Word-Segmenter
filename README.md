# Standard Tibetan Word Segmenter
 Training a neural net on a corpus w/ spaces so that it can find the word-boundaries in a corpus without them.

 The network takses as its inputs the embeddings of the syllables within its context window and uses these
 to predict whether the syllable at the center of the window is word-final or not.
