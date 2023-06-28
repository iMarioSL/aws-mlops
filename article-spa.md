# Poner un modelo en producción con AWS Lambda
Como científicos de datos somos capaces de aplicar complejos algoritmos de
Inteligencia Artificial para captar patrones de los datos y predecir el futuro.
Desafortunadamente, nuestros modelos solamente son _útiles_ si otras personas o
aplicaciones pueden _usarlos_ para realizar predicciones.

Poner un modelo en producción se refiere al proceso de disponibilizarlo a
usuarios alrededor del mundo. En este artículo construiremos un servicio de
predicción _serverless_ con el cual los usuarios podrán interactuar para
determinar si imágenes digitalizadas son cancerígenas o no.

> Puedes encontrar todos los recursos necesarios para replicar este proyecto [en
este repositorio](https://github.com/ArturoSbr/aws-mlops).

## Caso Práctico
Imagina un laboratorio médico que hace pruebas para detectar cancer de mama.
Actualmente, los doctores analizan visualmente las radiografías de los pacientes
para determinar la presencia de cáncer. La gerente de la clínica ha notado que
este proceso le toma mucho tiempo a los doctores, y te ha contratado a ti para
que desarrolles un modelo de inteligencia artificial que analice las imágenes.

Para esto, has propuesto lo siguiente:
1. El laboratorio tomará la prueba;
2. Los doctores enviarán la información a nuestro modelo; y
3. Nuestro modelo responderá con una predicción.

## ¿Por qué usar AWS Lambda?
In order to productionize a model back in the day, developers had to go to their
closest hardware store to buy a server, install it in their garage, host their
application in it and connect it to the internet so that their model could be
used by people all around the world. As you can imagine, this is a highly
expensive process, as it involves purchasing a server with the appropriate
amount of RAM, storage space, network card, cooling system, etc. On top of that,
developers had to worry about patching the server's operating system, updating
the dependencies used by the model, setting up firewalls to fend off hackers and
keeping an eye on the neighborhood kids to make sure no one tampered with the
hardware!

_Serverless_ is a business model where a vendor (i.e., Amazon Web Services)
owns and maintains the hardware needed to host an application and consumers can
use it on-demand to deploy software. Thanks to serverless solutions, we can
simply rent infrastructure from AWS and forget about buying a server,
maintaining its physical integrity, patching the operating system, etc.

AWS Lambda is the _crème de la crème_ of the serverless kingdom. It is a service
that allows us to write functions in our preferred programming language and
deploy them on servers owned and maintained by AWS. This means we do not have
to worry about provisioning or maintaining the instance that hosts the function.
All we need to do is write the function itself!

## Training a model
We need to build a model before we even worry about enabling it for online
consumption. Since the purpose of this article is learning how to deploy a
model, we will _speedrun_ the training portion of the process by executing a
Python script that outputs a scikit-learn classifier. To do this, download [the
repo](
    https://github.com/ArturoSbr/aws-mlops
), replicate the training environment and run the script.

```bash
% cd <path where you downloaded the repo>/aws-mlops # Set directory
% python3 -m venv my_venv # Create new virtual environment
% source my_venv/bin/activate # Activate it
% pip3 install -r requirements.txt # Install dependencies
% python3 code/fit-model.py` # Run script that exports model
```

Doing this will fit a gradient boosting classifier on the [breast cancer](
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
) dataset, which contains digitized images of cell nuclei that indicate if a
patient has breast cancer. If you want to dig deeper into how the model is fit,
you can check the notebook `code/fit-model.ipynb`, which does the exact same
thing as `code/fit-model.py` but has more comments explaining the process.

You should see a file called `clf.sav` in the `code/lambda-function/` directory.
The fitted model is stored in this file, and we can load it directly
in other scripts without having to repeat the training process all over again.

## Creating a Lambda Function
AWS Lambda allows us to host our own code on machines owned and maintained
by Amazon. From a developer's perspective, all we need to do is:
1. Writing a function in our preferred programming language;
2. Packing our dependencies into a deployment package; and
3. Deploying our Lambda function.

### 1. Writing the Lambda Function
If you take a look into the `code/lambda-function/` directory, you will see the
model that was exported by executing the `fit-model.py` script from the previous
section as well as a python script named `lambda_function.py`. As its name
suggests, the latter is the function we want to deploy in AWS Lambda.

Our goal is to enable an HTTP endpoint to our Lambda function so that the
clinic can use it to invoke it on demand. When the clinic sends a JSON file to
our function's endpoint, the event will trigger `lambda_function.py`, which will
load the classifier (named `clf`), receive the event sent by the clinic, get
its `'body'`, pass the values of the features to `clf` and return a prediction.

For example, if the clinic sends the following information in the `body` of
their request:
```
{
	"meanConcavePoints": 0.03821,
	"worstRadius": 14.97,
	"worstTexture": 24.64,
	"worstArea": 677.9,
	"worstConcavePoints": 0.1015
}
```
our function will respond with:
```
{
	"reason": "OK",
	"prediction": 0,
	"status": "200"
}
```

Our function can also handle errors. For example, if the clinic sends a request
that is missing a feature:
```
{
	"meanConcavePoints": 0.03821,
	"worstRadius": 14.97,
	"worstTexture": 24.64,
	"worstArea": 677.9
}
```
our function will respond with:
```
{
	"reason": "'body' must contain values for: meanConcavePoints, worstRadius, worstTexture, worstArea, worstConcavePoints",
	"prediction": "",
	"status": "400"
}
```

### 2. Creating a deployment package
To deploy our code as a Lambda function, we must pack our scripts and
dependencies into a _deployment package_. The idea is to pack all the resources
needed to run the function we wrote so that the machine that ultimately executes
our code has access to all the things it needs to run it as we would on our
local machines.

As you can see, `lambda_function.py` loads the file `clf.sav`, which is a
_pickled_ scikit-learn object. This means that our function requires
scikit-learn in the background. Unfortunately, packing scikit-learn alone will
not work because it requires other libraries:
```bash
% pip3 show scikit-learn
> Requires: joblib, numpy, scipy, threadpoolctl
```

The output tells us that in order to load scikit-learn, we must first load
joblib, numpy, scipy and threadpoolctl. Altogether, our deployment
package should contain:
1. The function (`lambda_function.py`);
2. The serialized model (`clf.sav`); and
3. The libraries stored in `my_venv/lib/` (joblib, numpy, threadpoolctl,
   scipy and scikit-learn).

Our deployment package must weigh 50 MB or less when zipped and 250 MB or less
when unzipped. In this example, the deployment package far-exceeds these limits,
so we must find a way to lighten our package.

#### Extra - AWS Lambda Layers
**NOTE:** You can skip this step if your deployment package meets the size
quotas.

[Lambda Layers](
    https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html
) allow us to store code and libraries _outside_ our Lambda functions. We can
add layers to our functions in order to reduce the size of our deployment
package. As a result, our functions can call the code stored in these layers
without directly containing them in their deployment packages!

To create a layer for a library:
1. Go to [PyPI](https://pypi.org/) and download the _manylinux_ wheel file of
   the library version required by your function (make sure it is compatible
   with the runtime and processor architecture of your Lambda function);
3. Unpack the library from your terminal using `% wheel unpack 
   <your-library.whl>`;
4. Change the name of the folder that was unpacked to _python_;
5. Zip the _python_ folder;
6. Sign in to AWS, select your region and go to _AWS Lambda_ > _Layers_ >
   _Create a Layer_;
7. Name your layer and upload the `python.zip` file; and
8. Click on _Create_.

For example, to create a numpy 1.24.1 layer for Python 3.9 and a 64-bit
processor, we must: 
1. Download the numpy 1.24.1 [wheel file](
numpy-1.24.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl) for
Python 3.9 and 64-bit processors;
2. Unpack the wheel file using `wheel unpack numpy-1.24.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`;
3. This will create a folder called _numpy-1.24.1_;
4. Rename this folder to _python_;
5. Zip the _python_ folder;
6. Login to AWS and go to _AWS Lambda_ > _Layers_ > _Create a Layer_;
7. Name it whatever you want and upload the `python.zip` file; and
8. Click on _Create_.

You must do this for all the libraries your function requires. Alternatively,
repeat this process only for the libraries that do not fit within the size
limit (though I recommend doing it for every library so that you can
individually add them to other Lambda functions in the future!).

Now that we have created the necessary Lambda layers, all we need in our
deployment package are the following files:
1. The function (`lambda_function.py`); and
2. The serialized model (`clf.sav`).

So compress these two files. In this example, we will compress them into a file
called `deployment-package.zip`.
```bash
cd code/lambda-function/ # Change directory
zip -r deployment-package.zip . # Compress everything in current directory
```

Now that our deployment package is ready, we are finally ready to create a
Lambda function.

### 3. Creating a Lambda Function
We will go to our AWS Management Console and search for _Lambda_. Make sure to
select the same region as where you stored your Lambda Layers! Now:

1. Go to _Functions_ > _Create function_ > _Author from scratch_;
2. Configure your function as follows:
   1. Name your function;
   2. Use a Python 3.9 runtime;
   3. select x86_64 as your architecture;
   4. leave _Permissions_ untouched;
   5. Click on _Advanced Settings_ > _Enable Function URL_ > _Auth type = NONE_.
3. Click on _Create function_.

![Lambda function configuration](
    https://drive.google.com/uc?export=view&id=1Bu271Wt5XNdu8Fl8rrUc6PtazUgZZckb
)

We just created a Lambda function with an open URL endpoint. Doing this will
automatically enable an HTTP endpoint to our function that anyone can connect
to! All they need in order to invoke the function is its URL. Your function
should look as follows:
![Lambda function URL](
    https://drive.google.com/uc?export=view&id=15CnA27KccCbC8HW5vluM5rYi_OCljXp4
)

In our example, the deployment package we just uploaded is missing scikit-learn
and all of its dependencies (because they are too large). To add them, scroll
all the way down and:
1. Click on _Add a layer_;
2. Click on _Custom layers_;
3. Select your joblib layer;
4. Select a version (there should only be one); and
5. Repeat for threadpoolctl, numpy, scipy and scikit-learn.

After doing this, your layers should look like this:

![Layers added to function](
    https://drive.google.com/uc?export=view&id=1Gioj4T6puagOmGl-SkE9LDRGIrdJELnH
)

## Posting requests to a Lambda function
Let's recap what we have done. We trained a model, exported it, wrote a
function that expects a request and passes it to the model to return a
prediction. We zipped the model and function into a deployment package and we
uploaded it to a Lambda function that has a Lambda URL endpoint as well as all
the Lambda Layers needed to load the model.

Namely, the Lambda URL that we enabled when creating the function is an HTTP
endpoint. We (or any application) can invoke the function by making calls to
this URL. In our business case, we want to allow the doctors to send information
to the model and have it respond with a prediction.

The doctors will do this by sending a POST request to the function's endpoint.
The doctor's request must contain the features expected by the model in the
request's `body`.

Imagine a doctor has the following scan:

| feature             | value   |
|---------------------|---------|
| Mean concave points | 0.07951 |
| Worst radius        | 24.86   |
| Worst texture       | 26.58   |
| Worst area          | 1886.0  |
| Mean concave points | 0.01    |

We will go over two examples to see how they could pass this information to the
model and receive a prediction.

### Example 1. Invoke the function with cURL
The doctor can send this request to the model using a terminal:
```bash
% curl -X POST \
      '{your-URL-here}' \
      -H 'Content-Type: application/json' \
      -d '{"meanConcavePoints": 0.07951, "worstRadius": 24.86, "worstTexture": 26.58, "worstArea": 1866.0, "worstConcavePoints": 0.01}'
```
Which responds with:
```
{"reason":"OK","prediction":1,"status":"200"}
```
This means that the observation we just sent to the model is not believed to be
malignant.

### Example 2. Invoke the function with Python
```python
# Import requests library
import requests

# Declare your function's url
url = "<your-function's-URL-here>"

# Declare observation
observation = {
    'meanConcavePoints': 0.07951,
    'worstRadius': 24.86,
    'worstTexture': 26.58,
    'worstArea': 1866.0,
    'worstConcavePoints': 0.01,
}

# Post request
req = requests.post(
    url=url,
    json=observation,
)

# Print response
print(req.json())
```
Which returns:
```python
{'reason': 'OK', 'prediction': 1, 'status': '200'}
```

### Example 3. Try sending a bad request
Just to be safe, let's see how our function handles bad requests. We will send
a request with only one feature.
```bash
% curl -X POST \
      '{your-URL-here}' \
      -H 'Content-Type: application/json' \
      -d '{"meanConcavePoints": 0.07951}'
```
Which responds with:
```
{"reason":"'body' must contain values for: meanConcavePoints, worstRadius, worstTexture, worstArea, worstConcavePoints","prediction":"","status":"400"}
```

## Conclusion
We have built a fully serverless prediction service using AWS Lambda, so
congratulations to you! In following this article, you have learned how to
create a deployment package, add external libraries as Layers, enable an HTTP
endpoint, and trigger your function from anywhere in the world.

Why is this useful? Remember that no matter how long we spend cross-validating
our performance metrics, the models we build are only useful insofar as other
users or applications can interact with them. In this sense, learning how to
productionize models with AWS Lambda is an invaluable skill that will reduce
your time-to-value by allowing you to productionize your code from the comfort
of your console.
