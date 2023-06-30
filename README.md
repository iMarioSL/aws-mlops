# Deploying a Machine Learning Model Using AWS Lambda
This repo trains a Machine Learning model and deploys it on AWS. The end result
is a fully serverless prediction service that users or applications can connect
to.

## Structure of this repo
```
.
├── code
│   ├── lambda-function
│   │   └── lambda_function.py
│   ├── fit-model.py    # ─┐ 
│   └── fit-model.ipynb # ─┴─ These files do the same thing. Run your favorite.
├── requirements.txt
└── README.md
````

## Running this repo
1. Create a virtual environment using `venv` and activate it.
2. Install dependencies (`$ pip install -r requirements.txt`).
3. Run either one of the `fit-model` files. This will export the model in
`./code/lambda-function/`.
    - `fit-model.py` is a simple Python script.
    - `fit-model.ipynb` is a notebook and has more context (but requires
    ipykernel).
4. Compress the `lambda-function/` directory (.zip).
5. Create an AWS Lambda function.
6. Create [Lambda
Layers](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html)
for `joblib`, `numpy`, `scipy` and `threadpoolctl` (read article for more info
details on how to do this!) and add them to your function.
7. Create a public [Lambda Function
ULR](https://docs.aws.amazon.com/lambda/latest/dg/lambda-urls.html).
8. Start posting requests!

## Examples
Example with `curl`:
```
curl -X POST \
    '<your-function-url-here> \
    -H 'Content-Type: application/json' \
    -d '{"meanConcavePoints": 0.07951, "worstRadius": 24.86, "worstTexture": 26.58, "worstArea": 1866.0, "worstConcavePoints": 0.1789}'
```

Example with [Insomnia](https://insomnia.rest/):
![Insomnia POST request](https://drive.google.com/uc?export=view&id=1IXNWLsN56oGmaUCYd3ix44OuqE-jY_7j)
