# AdaptiveTransformer
Barebones transformer decoder with [adaptive input](https://arxiv.org/pdf/1809.10853.pdf) and [adaptive softmax](https://arxiv.org/pdf/1609.04309.pdf).

The model is largely based on the pytorch [language model example code](https://github.com/pytorch/examples/tree/master/word_language_model). The transformer is trained on the wikitext-2 dataset by default.

```bash 
python main.py --cuda --epochs 6 --adsmax   # Train with adaptive softmax
python main.py --cuda --epochs 6 --adinp    # Train with adaptive input
python main.py --cuda --epochs 6 --tied_weights # Train with adaptive input and softmax with tied weights
```