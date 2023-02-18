# Deploying a Machine Learning Model Using AWS Lambda
This repo trains a Machine Learning model and deploys it on AWS. Te end result
is a fully serverless prediction service that users or applications can connect
to.

## How to run
The repo is structured as follows:
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

In order to run it:
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
7. Create a public [Lambda Function ULR]
(https://docs.aws.amazon.com/lambda/latest/dg/lambda-urls.html).
