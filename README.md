# HatCUP
## Prepare Requirements
- After download this project, you should first create the following folders: dataset, models and prediction

- Install cuda 11.x and cudnn

- Install python dependencies

  ```
  pip install -r requirements.txt
  ```

## Download Dataset

- Download the train, valid and test dataset from [here](https://mega.nz/folder/c0YXmSqQ#_bLG0IdrHR0zjBdRmBon-Q)
- put all the dataset files in folder 'dataset'

## Train

```
python HatCUP.py -data_path dataset/ -model_path models/model.pt --test_mode False
```

## Predict and Evaluate

```
python HatCUP.py -data_path dataset/ -model_path models/model.pt --test_mode True
```

After the model generate the predictions, you can run evaluation by running the scripts in `./eval`:

- For accuracy, recall, Meteor, SARI, run `eval.py`;
- For AED and RED, run `eval_edit_distance.py`;
- For GLEU, run `eval_gleu.py`; (this script should be run in an environment which contains both python2.7 and python3)

