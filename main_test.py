import glob
import os
import time

from src.load_visemenet import load
from src.infer_visemenet import infer
from src.create_dataset_csv import create_dataset_csv
from src.eval_viseme import eval_viseme

model_name='pretrain_biwi'
sess, model_dict = load(model_name)

# find all the audio files located under data/test_audio
test_audio_files = glob.glob("data/test_audio/*.wav")
test_audio_names = [os.path.basename(f) for f in test_audio_files]

for test_audio_name in test_audio_names:
    # convert audio wav to network input format
    start = time.time()
    create_dataset_csv(test_audio_name=test_audio_name)
    end = time.time()
    duration_create_dataset_csv = end - start

    # feedforward testing
    print(f'Processing "{test_audio_name}"')
    start = time.time()
    infer(sess, model_dict, test_audio_name=test_audio_name[:-4])
    end = time.time()
    duration_forward = end - start
    print(f'Finished predict "{test_audio_name}"')

    # output viseme parameter
    start = time.time()
    eval_viseme(test_audio_name[:-4])
    end = time.time()
    duration_eval_viseme = end - start
    print(f'Done "{test_audio_name}" Tcreate={duration_create_dataset_csv:.2f} Tforward={duration_forward:.2f} Teval={duration_eval_viseme:.2f}')
