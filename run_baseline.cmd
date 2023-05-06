@ECHO OFF

IF /I "%1"=="eval" GOTO eval

:train
python run_baseline.py --feature tfidf --analyzer char
python run_baseline.py --feature tfidf --analyzer word

python run_baseline.py --feature fasttext --analyzer char
python run_baseline.py --feature fasttext --analyzer word
python run_baseline.py --feature fasttext --analyzer sent

:eval
python run_baseline.py --eval
