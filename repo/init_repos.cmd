@ECHO OFF

REM QNLP-DisCoCat
git clone https://github.com/CQCL/qnlp_lorenz_etal_2021_resources

REM fastText
git clone https://github.com/facebookresearch/fastText
PUSHD fastText
pip install .
POPD

ECHO Done!
ECHO.

PAUSE
