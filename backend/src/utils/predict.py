import os

from src.utils.image_captioning import ImageCaptioning

models_path = os.getenv("MODELS_PATH", "src/models")

model_dir_eng = os.path.join(models_path, 'eng')
model_dir_ukr = os.path.join(models_path, 'ukr')

captioning_eng = ImageCaptioning(model_dir_eng)
captioning_eng.load_models()

captioning_ukr = ImageCaptioning(model_dir_ukr)
captioning_ukr.load_models()


def predict(file_bytes, language):
    if language == 'english':
        captioning = captioning_eng
    elif language == 'ukrainian':
        captioning = captioning_ukr
    else:
        raise ValueError("Unsupported language: {}".format(language))

    text, attention_plot = captioning.predict(file_bytes)
    return text, attention_plot
