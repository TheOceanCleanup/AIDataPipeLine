# This is a skeleton file that you can use to implement a training script. This
# includes the functionality required to train a model and register the
# run in Azure ML.
#
# To use this, look for the parts marked with TODO - these need to be adjusted
# or filled in.
# A logging object is available - whatever is logged to this will be registered
# in Azure ML as logs of the model training. Don't confuse this with run.log(),
# that is used to register statistics about the model, both parameters and
# performance metrics.
from azureml.core import Run
from utils import load_args, load_set_as_txt
from Tensorflow_YOLO.train import main as train_main
import logging
import shutil
import os

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
        ['weights', str, '.'],
        ['LR_INIT', float, 1e-4],
        ['LR_END', float, 1e-6],
        ['WARMUP_EPOCHS', int, 2],
        ['EPOCHS', int, 100]
    ])

    logger.debug(parameters.LR_INIT)
    logger.debug(parameters.LR_END)

    # Move weights file to correct folder
    for f in os.listdir(parameters.weights):
        shutil.copy(
            f"{parameters.weights}/{f}",
            f"Tensorflow_YOLO/model_data/{f}"
        )

    # Load/prepare the datasets
    train_set, train_labels = load_set_as_txt('train', parameters.train_sets)
    test_set, test_labels = load_set_as_txt('test', parameters.test_sets)

    #### Implement/perform model training ####

    logger.info("Starting training")

    try:
        mAP, fps = train_main(
            lr_init=parameters.LR_INIT,
            lr_end=parameters.LR_END,
            warmup_epochs=parameters.WARMUP_EPOCHS,
            epochs=parameters.EPOCHS
        )
    except Exception as e:
        logger.debug(e)
        logger.info("Retrying once")
        mAP, fps = train_main(
            lr_init=parameters.LR_INIT,
            lr_end=parameters.LR_END,
            warmup_epochs=parameters.WARMUP_EPOCHS,
            epochs=parameters.EPOCHS
        )

    logger.info("Finished training")

    #### Determine model performance ####
    logger.debug("Determining performance")

    # Performance is stored in mAP/results.txt, load this
    ap_cat = {}
    with open('mAP/results.txt') as f:
        for l in f.readlines():
            if '=' not in l or 'mAP' in l:
                continue
            else:
                logger.debug(l)
                l = l.split('=')
                ap = l[0].strip().rstrip('%')
                c = l[1].strip().split()[0]
                ap_cat[c] = ap

    #### Register model performance with the run ####

    # Load the run object
    run = Run.get_context()

    # Register the parameters used for training and the performance of the
    # model using run.log()
    logger.debug("Registering performance data")

    run.log('mAP', mAP)
    for k,v in ap_cat.items():
        run.log('AP_' + k, v)

    run.log('FPS', fps)

    # Also log the used parameters
    run.log('LR_INIT', parameters.LR_INIT)
    run.log('LR_END', parameters.LR_END)
    run.log('WARMUP_EPOCHS', parameters.WARMUP_EPOCHS)
    run.log('EPOCHS', parameters.EPOCHS)

    # Move the model files to the outputs/ folder. This is then automatically
    # attached to the Run, and to any models registered from that run
    logger.debug("Writing model data")

    logger.info("Train process finished")
