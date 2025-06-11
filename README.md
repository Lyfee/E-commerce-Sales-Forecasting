# E-commerce-Sales-Forecasting

## Install

Currently, this project has only been tested and confirmed to run correctly in a Python 3.10 environment.

```
git clone https://github.com/Lyfee/E-commerce-Sales-Forecasting
cd E-commerce-Sales-Forecasting
pip install -r requirements.txt
```

## Download Datasets

MYS:
- csv: [mys.csv](https://drive.google.com/file/d/1PLQlr4DCTc6tHu14yXYUqXYY63YSVsCX/view?usp=sharing)
- images: [mys-images.zip](https://drive.google.com/file/d/1faAl7HC2x9OwRml0HELf1-vhIESsLQYy/view?usp=sharing)

US:
- csv: [us.csv](https://drive.google.com/file/d/1JX-ujZzzoHXkWyotWxmKzbXP8RlWhIsB/view?usp=sharing)
- images: [us-images.zip](https://drive.google.com/file/d/1b3-kEfsptv5k-oOLm7k-YpDtoi5691En/view?usp=sharing)

UK:
- csv: [uk.csv](https://drive.google.com/file/d/1mpA-MPzkexhBSjMpFFNw6AOR06YO7yj-/view?usp=sharing)
- images: [uk-images.zip](https://drive.google.com/file/d/1a_dr-gW2XWNfwqALEAkyD2ZYNZzXE-h3/view?usp=sharing)

VN:
- csv: [vn.csv](https://drive.google.com/file/d/11Dtg4sKTgB6SardtGExF3zBQCYLdp-FR/view?usp=sharing)
- images: [vn-images.zip](https://drive.google.com/file/d/1Icu0JQUQJMHVM15wkW71Lu5J8D5s_hID/view?usp=sharing)

The data presented above has been cleaned from the raw collected data and can be directly used for model training. For an example of the raw data cleaning process, using the 'mys' region as a case study, please refer to the clean_data.ipynb file.

## Run Example

The following command will train a TFT model on the MYS dataset with a maximum encoder length of 20, maximum prediction length of 2, and a learning rate of 0.001. It will run five times to calculate the average model performance.

```
python3 tft.py --model tft --max_prediction_length 2 --max_encoder_length 20 --learning_rate 0.001 --train_batch_size 5 --dataset_path mys.csv --dataset_image_path mys-images
```

For other models and running parameters, please refer to the source code in arima.py, dl.py, and tft.py, and ensure you specify the correct dataset locations.