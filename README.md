# Usage
1) install cuda or cpu version of torch and set the DEVICE in the config accordingly
(transformers could potentially require a independent installation aswell)

2) set up app/config.py 

example:
'''
HOST = "0.0.0.0"
PORT = 8000

MONGO_DETAILS = "mongodb://admin:password@localhost:27017"

DELETION_POOLING_INTERVAL = 3600
EXPIRATION_TIME = 3600
BATCH_SIZE = 16
DEVICE = "cpu"
'''

3)  
```
pip install -r requirements.txt

docker compose up -d
python app/main.py
```