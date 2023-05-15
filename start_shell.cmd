@ECHO OFF
SET MODE_DEV=1
SET RAND_SEED=114514

REM use this script to startup deveopment env :)

IF /I "%1"=="py"     GOTO env_python
IF /I "%1"=="python" GOTO env_python

:env_cmd
CMD /K conda activate q
GOTO EOF

:env_python
CMD /K conda activate q ^& python -i -c "from run_quantum import * ; print('>> interactive python console ~')"
GOTO EOF

:EOF
