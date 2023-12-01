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
# The included environment.yaml installs latest TF V1 for linux with CUDA 10 and Python 3.7.16

conda create -f environment.yaml

# For Anaconda running on Windows, use the same command above but instead specify the environment_win.yml file:

conda create -f environment_win.yml


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

   The output is saved to a file of space-delimeted data values located at:

```
data/output_viseme/[your_audio_file_name]/mayaparam_viseme.txt
```

Each row of the output file represents an animation frame at a rate of 25 frames per second.

The data in the table are all fixed-point activation values represented to 4 decimal points
with possible values between 0.0000 and 1.0000

The meaning of each column is as follows:

| Column | Viseme active |                             FACS AUs |
| :----- | :-----------: | -----------------------------------: |
| 1      |      JAw      |                                   26 |
| 2      |      LIp      |                                    8 |
| 3      |     'AAH'     |                          16D 17A 27D |
| 4      |     'AAA'     | 10C 12C 14C 16D 17A     20B 22uB 27C |
| 5      |     'Eh'      |
| 6      |     'Ee'      |
| 7      |     'Ih'      |
| 8      |     'Oh'      |
| 9      |     'Uh'      |
| 10     |     'UUU'     |             17A 18D          27A 28A |
| 11     |     'Eu'      |
| 12     |    'Schwa'    |                                  27B |
| 13     |     'RRR'     |                 17A 18C     22C  27A |
| 14     |     'SSS'     |                  16E 17A         22C |
| 15     |     'SSH'     | 10B         16B     18C 20A 22C  27A |
| 16     |     'TTH'     | 10C             17A         22uD 27B |
| 17     |     'JY'      |
| 18     |    'LNTD'     |
| 19     |     'GK'      |
| 20     |     'MMM'     |                 17E         23uC 27A |
| 21     |     'FFF'     |             17D         23uA 27A 28B |
| 22     |  'WA_PEDAL'   |



## JALI animation in Maya

   * put your test audio file name in file 'maya_animation.py', line 4.
   * Then run 'maya_animation.py' in Maya with JALI environment to create talking face animation automatically. (If using different version of JALI face rig, the name of phoneme/co-articulation variable might varies.)
   * UPDATE: 'maya_animation.py' has been updated with the [public face rig](http://www.dgp.toronto.edu/~elf/jali.html) annotations. Feel free to play with it!

