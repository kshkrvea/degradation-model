# degradation-model

## Quick start
1. Clon repo
``` 
git clone https://github.com/ezgor/degradation-model
cd degradation-model
```
2. Do 
``` 
python3 run.py -m --model -i --input -o --output -b --block_size
```
Where 
```
usage: run.py [-h] [-m MODEL] [-i INPUT] [-o OUTPUT] [-b BLOCK_SIZE]

optional arguments:
  -h, --help            			Show this help message and exit
  -m MODEL, --model MODEL			Folder path to the pre-trained models
  -i INPUT, --input INPUT			Input folder
  -o OUTS, --output OUTPUT  			Output folder
  -b BLOCK_SIZE, --block_size BLOCK_SIZE	Size of block to split the image

```

# Acknowledgement
This implementation largely depends on [RealVSR](https://github.com/IanYeung/RealVSR). Thanks for the excellent codebase! 
