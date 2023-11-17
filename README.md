# VisemeNet Code Readme

## Environment

+ Python 3.7.16
+ Tensorflow 1.15.0 
+ Cudnn 10.0

## Python Package

+ numpy
+ scipy
+ python_speech_features

## Input/Output

+ Input audio needs to be 44.1kHz, 16-bit, WAV format
+ Output visemes are applicable to the JALI-based face-rig, see [HERE](http://www.dgp.toronto.edu/~elf/jali.html)

## JALI Viseme Annotation Dataset

+ BIWI dataset with well-annotated JALI viseme parameter. [[DATASET](https://www.dropbox.com/sh/oj13tvq9ggf2puz/AADBPyRUcyisFtKgCoDmNhLHa?dl=0)]   [[README](VisemeNet_Annotation_README.md)]

## At test time:


1. **Download this repository to your local machine:**

```
git clone https://github.com/gb96/VisemeNet_tensorflow.git  

cd VisemeNet_tensorflow 
```

2. **Create and install required envs and packages**

```
# The included environment.yaml installs latest TF V1 for linux with CUDA 10 and Python 3.7.
# You might also be able to get this working natively with conda under Windows.
conda create -n visnet python=3.7 -f environment.yaml


# activate conda environment
conda activate visnet
```

3. **Prepare data and model:**  

   * convert your test audio files into WAV format, put them in the directory data/test_audio/   
   * download the public face rig model from [HERE](https://www.dropbox.com/sh/7nbqgwv0zz8pbk9/AAAghy76GVYDLqPKdANcyDuba?dl=0), put all 4 files to data/ckpt/pretrain_biwi/  

4. **Forward inference:**  

   * Run command line
```
python main_test.py
```  
   The result locates at:  
```
data/output_viseme/[your_audio_file_name]/mayaparam_viseme.txt
```

5. **JALI animation in Maya:**

   * put your test audio file name in file 'maya_animation.py', line 4.
   * Then run 'maya_animation.py' in Maya with JALI environment to create talking face animation automatically. (If using different version of JALI face rig, the name of phoneme/co-articulation variable might varies.)
   * UPDATE: 'maya_animation.py' has been updated with the [public face rig](http://www.dgp.toronto.edu/~elf/jali.html) annotations. Feel free to play with it!

