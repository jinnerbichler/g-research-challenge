import requests
import zipfile
import io


TRAINING_DATA_URL = 'https://financialforecasting.gresearch.co.uk/data/train.csv.zip'
TEST_DATA_URL = 'https://financialforecasting.gresearch.co.uk/data/test.csv.zip'

r = requests.get(TEST_DATA_URL)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()
