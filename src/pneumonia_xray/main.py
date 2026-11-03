from .modules import launch_learning,launch_predict
from .config import EPOCHS, LEARNING_RATE, PATH_TO_DATA, PATH_TO_LAST_MODEL_CHECKPOINT, IMAGE_SIZE, CLASSNAMES, PATH_TO_PREDICTION_DATA

import sys

if __name__ == "__main__":

    try:
        action = sys.argv[1]
    except IndexError:
        print("Il faut au moins une action en argument (train ou predict)")
        sys.exit()

    if action=="train":
        launch_learning(epochs=EPOCHS, learning_rate=LEARNING_RATE, path_to_data=PATH_TO_DATA, path_to_save_last=PATH_TO_LAST_MODEL_CHECKPOINT)
    
    elif action=="predict":
        launch_predict(image_size=IMAGE_SIZE, images_path=PATH_TO_PREDICTION_DATA, classnames=CLASSNAMES, path_to_load=PATH_TO_LAST_MODEL_CHECKPOINT)

