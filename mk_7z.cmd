7z a -t7z ^
  -mx9 ^
  -xr!*.7z ^
  -xr!ref\*.pptx ^
  -xr!ref\*.*.pdf ^
  -xr!repo\ ^
  -xr!data\*.bin ^
  -xr!data\*_cleaned.csv ^
  -xr!data\*_tokenized.txt ^
  -xr!data\valid.csv ^
  -xr!data\unknown.csv ^
  -xr!log\ ^
  -xr!tmp\ ^
  -xr!.git\ ^
  -xr!.vscode\ ^
  -xr!__pycache__\ ^
  YouroQNet.7z ^
  .

7z u -t7z ^
  -ir!repo\*.cmd ^
  YouroQNet.7z
