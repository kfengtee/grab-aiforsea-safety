# Submission For "AI For SEA: Safety"

<p align=center>
    <a href="#readme">
        <img alt="logo" width="30%" src="https://static.wixstatic.com/media/397bed_e0fd4340ff5f40de876b26f0fb7e1f83~mv2.png/v1/fill/w_458,h_458,al_c,q_80,usm_0.66_1.00_0.01/Grab%20EDM_Safety.webp">
    </a>
</p>

This is a machine learning model that can detect dangerous driving using telematics data collected during the trip. More details about this project can be found at https://www.aiforsea.com/safety. 

## About The Repository
1. Complete documentation of this project: **documentation.ipynb**
2. Python library to predict dangerous driving: **predict_model.py**
3. Pre-trained model weights directory: **model_weights/**
4. Demonstratation of python library: **demo.ipynb**
5. Reusable scripts (Model searching tools): **utils.ipynb**

## Highlights
1. A total of **52 features** are generated from raw telematics data using three approaches: <br>
* **Approach 1**: Statistical summary of telematics data <br>
* **Approach 2**: Count outlying driving behaviours based on telematics readings <br>
* **Approach 3**: Sliding windows aggregated features 
2. Six blended (stacked) models are used in the prediction pipeline, which consists of 3 layers: <br>
* **Layer 1**: 2 weak learners <br>
* **Layer 2**: 3 strong learners <br>
* **Layer 3**: 1 meta learner
3. The model performance on self-define hold-out test dataset achieved:
* **ROC Score**: 0.7513
* **Accuracy Score**: 0.7853

**NOTE: Please refer to *documentation.ipynb* for more details about this project.**

## Notes To Evaluators
(Tested on Python Version: 3.7)

First, clone the repository and install the dependencies.
```sh
git clone https://github.com/kfengtee/grab-aiforsea-safety.git
cd grab-aiforsea-safety
pip install -r requirements.txt
```

Then, upload your hold-out test set (raw telematics data and labels) to this repository.

To use the library, you can do the following:
```python
import predict_model

classifier = predict_model.DangerousDrivingClassifier('model_weights') # load the pre-trained weights

# replace "dir_to_raw_telematics_data" with actual hold-out test data directory
output = classifier.predict("dir_to_raw_telematics_data") # output: DataFrame, columns = ['bookingID', 'prob', 'label']
```
Please look at **demo.ipynb** for more demonstration details. 

License
----
MIT

