import autokeras as ak
import tensorflow as tf
import os
from pathlib import Path

HERE = Path(__file__).parent.absolute()

train_data = ak.image_dataset_from_directory(
    HERE.joinpath('traindata', 'digits'),
    image_size=(120, 120),
    batch_size=16)

clf = ak.ImageClassifier(overwrite=True, max_trials=1, project_name='digitsTrainer')
clf.fit(train_data, epochs=5)
print(clf.evaluate(train_data))

clf.export_model().save("digits", save_format="tf")

# TODO: use tesseract to verify images for traindata and train on many more digits