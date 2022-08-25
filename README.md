"# VINIF_IdentifyPerson" 
Example for testing on Physionet
## Install and download dataset
Create directory/ folder for dataset
```bash
/Users/"your current user"/mne_data
```
install required packages in **requirement.txt**
## Run example
```bash
python trainPhisio.py  --modelName CNN --output ./res/thinkingPhy.txt --bandL 4 --bandR 8 --thinking False --trainTestSeperate False --windowSize 128  --eaNorm EA
```
