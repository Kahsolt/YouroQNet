@ECHO OFF

REM classic methods
git clone https://github.com/facebookresearch/fastText
git clone https://github.com/649453932/Chinese-Text-Classification-Pytorch
git clone https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch
git clone https://github.com/FreedomIntelligence/TextClassificationBenchmark
git clone https://github.com/linguishi/chinese_sentiment
git clone https://github.com/cjymz886/text-cnn
git clone https://github.com/PracticingMan/chinese_text_cnn
git clone https://github.com/gaussic/text-classification-cnn-rnn
git clone https://github.com/HappyShadowWalker/ChineseTextClassify

REM quantum methods
git clone https://github.com/CQCL/qnlp_lorenz_etal_2021_resources
git clone https://github.com/ICHEC/QNLP
git clone https://github.com/rdisipio/qnlp
git clone https://github.com/mullzhang/quantum-nlp
git clone https://github.com/ankushgpta2/Quantum_NLP
git clone https://github.com/IChowdhury01/Quantum-NLP-Sentence-Understanding
git clone https://github.com/AkimParis/quantumNLP_jp
git clone https://github.com/helloerikaaa/quweeting
git clone https://github.com/HalaBench/Quantum_ML
git clone https://github.com/Levyya/ComplexQNN

REM tiny-q as scaffold
git clone https://github.com/Kahsolt/Tiny-Q
PUSHD Tiny-Q
python setup.py install
POPD

ECHO Done!
ECHO.

PAUSE
