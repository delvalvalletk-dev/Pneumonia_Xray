PATH_TO_DATA = "C:/Users/Killian/Documents/github/Programmation/Python/Pneumonia_Xray/data/chest_xray"

IMAGE_SIZE = [224,224]
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
SEED = 42
CLASSNAMES = {1 : "PNEUMONIE" , 0 : "NORMAL"}

PATH_TO_BEST_MODEL_CHECKPOINT = "C:/Users/Killian/Documents/github/Programmation/Python/Pneumonia_Xray/model_state/best.pt"
PATH_TO_LAST_MODEL_CHECKPOINT = "C:/Users/Killian/Documents/github/Programmation/Python/Pneumonia_Xray/model_state/last.pt"

PATH_TO_PREDICTION_DATA = "C:/Users/Killian/Documents/github/Programmation/Python/Pneumonia_Xray/data/image_test"
PREDICTION_BATCH_SIZE = 5