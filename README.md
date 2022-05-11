# LinguisticStructureLM
Transformer-based language modeling with symbolic linguistic structure representations. To be published at NAACL 2022 as "Linguistic Frameworks Go Toe-to-Toe at Neuro-Symbolic Language Modeling" by Jakob Prange, Nathan Schneider, and Lingpeng Kong.

To install dependencies, run:
`pip install -r requirements.txt`

To reproduce the main results (table 2 in the paper), complete the following steps:
1. Obtain evaluation data. We used the annotated and [MRP](http://mrp.nlpl.eu/2020/index.php)-formatted WSJ data, so we cannot release it here. Contact me to discuss details (you'll probably need an LDC license).
2. Download the [trained models](https://drive.google.com/drive/folders/1U1uvIgkVLS-kBrkRPPGE7iywpY7W9Yx_?usp=sharing) into this directory
3. Edit `lm_eval.sh` to match your local environment
4. Run: `sh eval_all_lm.sh`
5. The results will be written to `stdout` by the eval.py, which will be collected in a file called `eval-dm,dm,psd,eds,ptg,ud,ptb-phrase,ptb-func-10-0001-0.0_0.0-0-14combined.out` by lm_eval.sh
6. Run: `cat SemanticGraphToText/uos/eval-dm,dm,psd,ptg,eds,ud,ptb-phrase,ptb-func-10lm-0001-0.0_0.0-0-14,-10-0001-0.0_0.0-0-14combined.out | grep ";all;" | grep gold`, which will give you a bunch of semicolon-separated lines you can paste into your favorite spreadsheet. Voila!

To evaluate a trained model more generally (might require additional input file; contact me!), edit `lm_eval.sh` to match your environment and directory structure, uncomment the lines you want in `eval_all_lm.sh` and run:
`sh eval_all_lm.sh SEED` where `SEED` is the last number before `.pt` in the model name (currently only seed=14 models are available for download).

To train a new model (requires access to .mrp-formatted and preprocessed data, which you can find [here](http://mrp.nlpl.eu/2020/index.php) and/or contact me about), edit `lm.sh` to match your environment and directory structure, uncomment the lines you want in `run_all_lm.sh` and run:
`sh run_all_lm.sh SEED` where `SEED` is a custom random seed you can set.

To get more info on commandline arguments, run:
`python train.py` or `python eval.py`
 
