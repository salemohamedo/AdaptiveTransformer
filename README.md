# AdaptiveTransformer
Barebones transformer decoder with [adaptive input](https://arxiv.org/pdf/1809.10853.pdf) and [adaptive softmax](https://arxiv.org/pdf/1609.04309.pdf).

The model is largely based on the pytorch [language model example code](https://github.com/pytorch/examples/tree/master/word_language_model). The transformer is trained on the wikitext-2 dataset by default.

```bash 
python main.py --cuda --epochs 6 --adsmax   # Train with adaptive softmax
python main.py --cuda --epochs 6 --adinp    # Train with adaptive input
python main.py --cuda --epochs 6 --tied_weights # Train with adaptive input and softmax with tied weights
```
Wikitext-2 results for 6 training epochs

|Model                     | Total Train Time (s) | Test Perplexity  |
|---------------------|----------------------|------------------|
| Vanilla             | 211.3                | 195.85           |
| Adaptive Softmax    | 122.4                | 203.71           |
| Adaptive Softmax (Pytorch Impl)   | 118.4                | 206.71           |
| Adaptive Input      | 248.1                | 218.91           |
| Adaptive Input (FAIR Impl)     | 249.2                | 240.39        |
| Both (Tied Weights) | 150.9                | 215.99           |

Text8 results for 6 training epochs

|Model                     | Total Train Time (s) | Test Perplexity  |
|---------------------|----------------------|------------------|
| Vanilla             | 8592                | 409.36           |
| Adaptive Softmax    | 1734                | 417.51           |
| Adaptive Input      | 8466                | 418.64           |
| Both (Tied Weights) | 1560                | 404.58           |