from .modules import launch_learning

from .config import EPOCHS, LEARNING_RATE, PATH_TO_DATA, PATH_TO_LAST_MODEL_CHECKPOINT

if __name__ == "__main__":
    launch_learning(epochs=EPOCHS, learning_rate=LEARNING_RATE, path_to_data=PATH_TO_DATA, path_to_save_last=PATH_TO_LAST_MODEL_CHECKPOINT)



