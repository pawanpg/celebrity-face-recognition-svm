# Celebrity Face Recognition (Haarcascade + Wavelets + ML)

Minimal face recognition/classification pipeline using:
- **OpenCV** (Haar cascades) for face detection
- **PyWavelets (Wavelet transform)** for feature extraction
- **scikit-learn** (SVM, RandomForest, LogisticRegression) for classification
- **Flask** for serving predictions
- **pandas / numpy** for data handling


## Project layout

├── model/
│ ├── sports_celebs_classfication.py # Training and classification code
│ ├── dataset/ # Dataset (raw + cropped images)
│ ├── opencv/haarcascades/ # Haarcascade XML files
│ └── test_images/ # Sample test images
│
├── server/
│ ├── server.py # Backend service for predictions
│ ├── util.py # Helper utilities
│ ├── wavelet.py # Wavelet feature extraction
│ └── artifacts/ # Saved model & metadata
│
├── requirements.txt
├── README.md

## Quickstart
pip install -r requirements.txt

## Train Model
cd model
python sports_celebs_classfication.py

## Run Server
cd server
python server.py


**Results**
With the sample dataset (Virat Kohli, Lionel Messi, Roger Federer, Maria Sharapova, Serena Williams):

Training accuracy: ~85-95% (depending on model)
Validation accuracy: ~80-85% (varies with dataset size/quality)
For better results, add more high-quality cropped images to model/dataset/cropped/.

## Notes
Add your dataset into model/dataset/ with subfolders per celebrity.
Delete/clean bad images manually if detection failed.
This project is intended for educational/demo purposes.