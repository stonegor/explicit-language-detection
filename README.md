# Usage
1) Unpack model into classifier/model

2) install cuda or cpu version of torch and set the DEVICE in the config accordingly
(transformers could potentially require a independent installation aswell)

3)  ```
pip install -r requirements.txt

docker compose up -d
python app/main.py
```