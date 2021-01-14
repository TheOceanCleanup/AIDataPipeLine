from azureml.core import Run, Dataset
import subprocess
import logging
import os
import shutil
from utils import load_args, find_set, save_set_as_tfrecords, DATASTORE_NAME
import select
import re
import sys

#from tf2od.model_main_tf2 import tf
import tensorflow as tf
tf.debugging.set_log_device_placement(True)


logger = logging.getLogger('model')
fh = logging.FileHandler('logs/model.log')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
fh.setLevel('DEBUG')
logger.addHandler(fh)
logger.setLevel('DEBUG')


if __name__ == "__main__":
    # Parse arguments. The arguments contains the (path to the) data and the
    # parameters to the model. Provide the parameters as
    # [<name>, <type>, <default value>]
    parameters = load_args([
        ['model_name', str, ''],
        ['num_train_steps', int, 1002],
        ['sample_1_of_n_eval_examples', int, 1],
        ['checkpoint_files', str, '.'],
        ['config_num', str, '']
    ])

    # Move checkpoint files to this folder
    logger.debug(os.listdir(parameters.checkpoint_files))
    os.makedirs('tf2od/pretrained_model')
    for f in os.listdir(f"{parameters.checkpoint_files}/"):
        if f.startswith(parameters.model_name):
            logger.debug(f)
            shutil.copy(
                f"{parameters.checkpoint_files}/{f}",
                f"tf2od/pretrained_model/{f}"
            )
    logger.debug(os.listdir('.'))

    # Generate as CSV, list of labels as labelmap.pbtxt
    train_path = save_set_as_tfrecords('train', parameters.train_sets, 'tf2od/annotations')
    print(train_path)
    test_path = save_set_as_tfrecords('test', parameters.test_sets, 'tf2od/annotations')
    print(test_path)

    #### Implement/perform model training ####

    logger.info("Starting training")

    # Define subprocesses
    init_processes = [[
        "python",
        "model_main_tf2.py",
        "--model_dir=../logs/",
        f"--pipeline_config_path=./configs/{parameters.model_name}{parameters.config_num}.config",
        f"--eval_on_train_data=True",
        f"--num_train_steps={parameters.num_train_steps}",
        f"--sample_1_of_n_eval_examples={parameters.sample_1_of_n_eval_examples}",
        f"--checkpoint_every_n=1250"
    ],[
        "python",
        "model_main_tf2.py",
        "--model_dir=../logs/",
        f"--pipeline_config_path=./configs/{parameters.model_name}{parameters.config_num}.config",
        f"--checkpoint_dir=../logs/",
        f"--sample_1_of_n_eval_examples=1",
        "--eval_timeout=1800",
    ],[
        "python",
        "model_main_tf2.py",
        "--model_dir=../logs/",
        f"--pipeline_config_path=./configs/{parameters.model_name}{parameters.config_num}.config",
        f"--checkpoint_dir=../logs/",
        f"--sample_1_of_n_eval_examples=1",
        "--eval_timeout=1",
    ]]

    # Run train and evaluate process
    p0 = subprocess.Popen(init_processes[0], cwd = "tf2od",
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        encoding="utf-8")
    p1 = subprocess.Popen(init_processes[1], cwd = "tf2od",
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        encoding="utf-8")

    # Print and save output
    pp = [p0.stdout, p1.stdout]
    stdout = []
    while True:
        rstreams, _, _ = select.select(pp, [], [])
        for stream in rstreams:
            line = stream.readline().strip()
            if len(line) > 0:
                print(line)
            stdout.append(line)
            if ('CUDA_ERROR_OUT_OF_MEMORY' in line)  or ('Resource exhausted: OOM when allocating tensor with shape' in line):
                sys.exit()
        if (p0.poll() != None):
            rstreams, _, _ = select.select(pp, [], [])
            for stream in rstreams:
                line = stream.readline().strip()
                if len(line) > 0:
                    print(line)
                stdout.append(line)
            break
    p1.terminate()

    logger.info("Finished training")
    #### Determine model performance ####
    logger.debug("Measuring performance")

    # Run final evaluate process
    p2 = subprocess.Popen(init_processes[2], cwd = "tf2od",
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        encoding="utf-8")

    # # Print and save output
    while True:
        line = p2.stdout.readline().strip()
        print(line)
        stdout.append(line)
        if ('Loss/total_loss:' in line)  or ('Timed-out waiting for a checkpoint' in line):
            p2.terminate()
            break

    # # Convert output to single big string
    stdout = " ".join(stdout)

    #### Regex to extract results
    re_mAP5095 = re.compile('Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \] = ([0-9]+\.[0-9]+)')
    re_mAP50 = re.compile('Average Precision  \(AP\) @\[ IoU=0.50      \| area=   all \| maxDets=100 \] = ([0-9]+\.[0-9]+)')
    re_localization_loss = re.compile('INFO:tensorflow:	\+ Loss/localization_loss: ([0-9]+\.[0-9]+)')
    re_regularization_loss = re.compile('INFO:tensorflow:	\+ Loss/regularization_loss: ([0-9]+\.[0-9]+)')
    re_classification_loss = re.compile('INFO:tensorflow:	\+ Loss/classification_loss: ([0-9]+\.[0-9]+)')
    re_total_loss = re.compile('INFO:tensorflow:	\+ Loss/total_loss: ([0-9]+\.[0-9]+)')

    # Load the run object & Register model performance with the run ####
    run = Run.get_context()
    for reg, log_name in zip([re_mAP5095, re_mAP50, re_localization_loss,
                              re_regularization_loss, re_classification_loss, re_total_loss],
                              ['mAP@0.5:0.95', 'mAP@0.5', 'val_loc_loss',
                              'val_reg_loss', 'val_class_loss', 'val_total_loss']):
        f = [float(r) for r in reg.findall(stdout)]
        if len(f) > 0:
            run.log_list(log_name, f)

    logger.debug("Done measuring performance")


    # Register the parameters used for training and the performance of the
    # model using run.log()
    logger.debug("Registering performance data")

    run.log('num_train_steps', parameters.num_train_steps)
    # run.log('sample_1_of_n_eval_examples', sample_1_of_n_eval_examples)

    # Write the model file to the outputs/ folder. This is then automatically
    # attached to the Run, and to any models registered from that run
    # logger.debug("Writing model data")

    # Export models

    os.makedirs('outputs', exist_ok = True)
    os.makedirs('outputs/detections', exist_ok = True)
    p = subprocess.run(
        [
            "python",
            "exporter_main_v2.py",
            "--input_type=image_tensor",
            f"--pipeline_config_path=./configs/{parameters.model_name}.config",
            f"--trained_checkpoint_dir=../logs/",
            f"--output_directory=../outputs/"
        ],
        cwd="tf2od"
    )

    # Make predictions
    #fps = detect(parameters.test_sets)
    #run.log("FPS", fps)

    #detect(parameters.test_sets)
    logger.info("Train process finished")
    #shutil.copytree('tf2od/annotations', 'outputs/annotations')
