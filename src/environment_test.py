"""Check that the classical ML environment imports correctly.

Run from the project root, with the virtual environment activated:

    python src/environment_test.py

Expected output: package versions, then a success message.
"""

import numpy as np
import pandas as pd
import sklearn
import cv2
import PIL

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("OpenCV:", cv2.__version__)
print("Pillow:", PIL.__version__)

print("Environment working successfully")
