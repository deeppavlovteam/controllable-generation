To run experiments you should have python3.7 on your machine.
To run generation of responses with Blenderbot you should install requirements:

```
    pip install -r requirements.txt
```
You should download weights of the Blendernot with PALs (blenderbot_pals.pth.tar) from https://drive.google.com/file/d/1KF5cPDjxrpfyoZo3GojfNG7P-LyLsxNf/view?usp=sharing and place to the same directory as scripts and Daily Dialog test set https://drive.google.com/file/d/11b022IUiW5CSC0XTsOwf0avoRz7Xnjx9/view?usp=sharing.

The file with test samples of Dialy Dialog (daily\_dialog\_da\_sent\_val.tsv) should be in the same directory as scripts.

Generation of responses with Blenderbot with PALs on Daily Dialog dataset:

```
    python3 generate\_blenderbot\_cont.py -d <gpu\_num> -out <responses\_filename>
```

gpu\_num is the ordinal of GPU you want to use for running the Blenderbot.

output\_filename is the name of file where the generated responses will be written. This file should be used for scores calculation with calculate\_scores.py script.

Generation of responses with Blenderbot 400M from Hugginface library:

```
    python3 generate\_blenderbot\_400M.py -d <gpu\_num> -out <responses\_filename>
```

Generated responses for different types of Blenderbot models (controllable, 400M, 90M) should be written in different files.

Calculation of control accuracy:

```
    python3 calculate\_scores.py -d -in <responses\_filename>
```

To calculate control accuracy you should download weights for dialog acts classifier from https://drive.google.com/file/d/1TFVwWQOWFZHCO6fsaip09U31XaWPC3Sq/view?usp=sharing and unpack them into ~/.deeppavlov/models/classifiers/dialog\_acts\_hist and weights for sentiment classifier from https://drive.google.com/file/d/13igJJVW1JoqOgW-zB3ABxKv_AqWvEY7V/view?usp=sharing and unpack into ~/.deeppavlov/models/classifiers/sentiment\_hist.

<responses\_filename> is filename with generated responses.

You should place the file with Wizard of Wikipedia test set, annotated with dialog acts and sentiment, (wow\_da\_sent\_test.json, download from https://drive.google.com/file/d/11aAeviW9JS-EsjBBp9c8BSz_rBLBDRnu/view?usp=sharing) in the same folder as the scripts for perplexity calculation.

Calculation of perplexity of controllable Blenderbot:
```
    python3 perplexity\_cont\_gk.py
```

Calculation of perplexity of Blenderbot 400M:
```
    python3 perplexity\_400M\_gk.py
```

Calculation of question asking accuracy:

```
    python3 question_asking.py -in <responses\_filename>
```
