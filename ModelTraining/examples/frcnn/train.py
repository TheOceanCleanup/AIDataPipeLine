from azureml.core import Run, Dataset
import subprocess
import logging
import os
import shutil
from utils import load_args, find_set, load_set_as_csv_pbtxt, DATASTORE_NAME

from frcnn.model_main_tf2 import tf

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
        ['num_train_steps', int, 10000],
        ['sample_1_of_n_eval_examples', int, 1],
        ['checkpoint_dataset', str, '.']
    ])

    num_train_steps = parameters.num_train_steps
    sample_1_of_n_eval_examples = parameters.sample_1_of_n_eval_examples

    # Move checkpoint files to this folder
    logger.debug(os.listdir(parameters.checkpoint_dataset))
    for f in os.listdir(f"{parameters.checkpoint_dataset}/"):
        logger.debug(f)
        shutil.copy(
            f"{parameters.checkpoint_dataset}/{f}",
            f"./{f}"
        )

    logger.debug(os.listdir('.'))

    # Generate as CSV, list of labels as labelmap.pbtxt
    train_set, train_labels = load_set_as_csv_pbtxt('train', parameters.train_sets, True)
    test_set, _ = load_set_as_csv_pbtxt('test', parameters.test_sets, False)

    #### Implement/perform model training ####

    logger.info("Starting training")

    # Train the model

    p = subprocess.run(
        [
            "python",
            "model_main_tf2.py",
            "--model_dir=../outputs/",
            f"--num_train_steps={num_train_steps}",
            f"--sample_1_of_n_eval_examples={sample_1_of_n_eval_examples}",
            "--pipeline_config_path=pipeline.config",
            "--alsologtostderr"
        ],
        cwd="frcnn",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    logger.debug(p.stdout)
    logger.warning(p.stderr)

    logger.info("Finished training")

    #### Determine model performance ####
    logger.debug("Measuring performance")

    logger.debug("Done measuring performance")

    #### Register model performance with the run ####

    # Load the run object
    run = Run.get_context()

    # Register the parameters used for training and the performance of the
    # model using run.log()
    logger.debug("Registering performance data")

    run.log('num_train_steps', num_train_steps)
    run.log('sample_1_of_n_eval_examples', sample_1_of_n_eval_examples)

    # Write the model file to the outputs/ folder. This is then automatically
    # attached to the Run, and to any models registered from that run
    # logger.debug("Writing model data")

    logger.info("Train process finished")
