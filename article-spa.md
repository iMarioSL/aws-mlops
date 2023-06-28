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

## ¿Por qué AWS Lambda?
Unos años atrás, para poner un modelo en producción, los desarrolladores tenían
que comprar un servidor, instalarlo en su garage, desplegar su aplicación en él
y darle acceso a internet para que el mundo conectarse con este. Como podrás
imaginar, solía ser un proceso muy caro, pues involucraba comprar un servidor
con la RAM, almacenamiento, tarjeta de red y sistema de enfriamiento adecuados.
Además, los desarrolladores tenían que actualizar el sistema operativo del
servidor, defenderlo de ciberataques así como proteger su integridad física.

_Serverless_ es un modelo de negocio donde un provedor (por ejemplo, Amazon Web
Services) le da mantenimiento al hardware necesario para desplegar una
aplicación y los consumidores pueden usarlo cuando ellos quieran. Gracias a este
tipo de soluciones, ahora podemos rentar el hardware de AWS y solamente nos
preocupamos por escribir código.

En este sentido, AWS Lambda es el servicio líder del reino _serverless_, pues es
un servicio que nos permite escribir funciones en nuestro lenguaje de
programación preferido y desplegarlo en hardware de AWS. Esto significa que no
tenemos que preocuparnos por provisionar o darle mantenimiento a la instancia
en donde nuestra función está hospedada. Lo único que tenemos que hacer es
escribir la función en sí.

## Entrenar un modelo
El proceso de poner un modelo en producción está precedido por la etapa de
entrenamiento del mismo. Dado que el proósito de este artículo es aprender a
produccionalizar un modelo, vamos a pasar rápidamente por el proceso de
entrenamiento. Para esto, descarga [el repositorio](
    https://github.com/ArturoSbr/aws-mlops
), replica el ambiente de Python y ejecuta el script.

```bash
% cd <path where you downloaded the repo>/aws-mlops # Cambia de directorio
% python3 -m venv my_venv # Crea un nuevo ambiente virtual
% source my_venv/bin/activate # Actívalo
% pip3 install -r requirements.txt # Installa las dependencias
% python3 code/fit-model.py` # Corre el script que exporta un modelo entrenado
```

Estos comandos entrenarán un clasificador de potenciación de gradiente usando el
[conjunto de datos de cáncer de mama](
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
) el cual contiene imágenes digitalizadas de células que indican si un paciente
tiene cáncer. Si deseas averiguar más acerca de cómo se entrenó el modelo, te
sugiero revisar el cuaderno `code/fit-model.ipynb`, el cual hace exáctamente lo
mismo que `code/fit-model.py` pero tiene comentarios adicionales.

Tras ejecutar el código de Python, deberías de ver un archivo llamado `clf.sav`
en el directorio `code/lambda-function/`. El modelo entrenado está guardado en
este archivo y lo podemos cargar directamente en otros archivos sin tener que
repetir el proceso de entrenamiento de nuevo.

## Crear una función Lambda
AWS Lambda nos permite hospedar una función en computadoras que son propiedad de
Aamazon. Desde el punto de vista de un desarrollador, lo único que tenemos que
hacer es:
1. Escribir una función en nuestro lenguaje de programación preferido;
2. Empacar nuestras dependencias; y
3. Desplegar nuestra función.

### 1. Escribir la función Lambda
En el directorio `code/lambda-function/` verás el modelo que exportamos al
ejecutar el archivo `fit-model.py` así como un script llamado
`lambda_function.py`. Como su nombre lo indica, el segundo archivo representa
la función que queremos desplegar en AWS Lambda.

En el contexto del caso de negocio, nuestro objetivo es habilitar un extremo
HTTP (HTTP endpoint) para que la cínica pueda invocar al modelo remotamente.
Queremos que los doctores puedan enviar un archivo JSON a este extremo para
activar el archivo `lambda_function.py`, el cual cargará el modelo (en un objeto
llamado `clf`), recibirá el evento enviado por la clínica, extraerá el cuerpo
(`body`) del mensaje, le pasará los atributos al objeto `clf` y regresará una
predicción.

Por ejemplo, si la cínica manda la siguiente información en el cuerpo del
evento:
```
{
    "meanConcavePoints": 0.03821,
    "worstRadius": 14.97,
    "worstTexture": 24.64,
    "worstArea": 677.9,
    "worstConcavePoints": 0.1015
}
```
Nuestra función responderá con:
```
{
    "reason": "OK",
    "prediction": 0,
    "status": "200"
}
```

Nuestra función también puede manejar errores. Por ejemplo, si la clínica envía
un evento con un atributo faltante:
```
{
    "meanConcavePoints": 0.03821,
    "worstRadius": 14.97,
    "worstTexture": 24.64,
    "worstArea": 677.9
}
```
Nuestra función responderá:
```
{
    "reason": "'body' must contain values for: meanConcavePoints, worstRadius, worstTexture, worstArea, worstConcavePoints",
    "prediction": "",
    "status": "400"
}
```

### 2. Crear un paquete de despliegue (_deployment package_)
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
