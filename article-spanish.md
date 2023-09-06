# Poner un modelo en producción con AWS Lambda
Como científicos de datos somos capaces de aplicar complejos algoritmos de
Inteligencia Artificial para captar patrones y predecir el futuro.
Desafortunadamente, nuestros modelos solamente son _útiles_ si otras personas o
aplicaciones pueden _usarlos_ para realizar predicciones.

Poner un modelo en producción se refiere al proceso de disponibilizarlo a
usuarios alrededor del mundo. En este artículo usaremos AWS Lambda para
produccionalizar un modelo capaz de detectar enfermedades en pruebas de
pacientes.

> Puedes encontrar todos los recursos necesarios para replicar este proyecto [en
este repositorio](https://github.com/ArturoSbr/aws-mlops).

## Caso Práctico
Imagina un laboratorio médico que hace pruebas para detectar cáncer de mama.
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
y darle acceso a internet para que el mundo pudiera conectarse con este. Como
podrás imaginar, solía ser un proceso muy caro, pues involucra comprar un
servidor con la RAM, almacenamiento, tarjeta de red y sistema de enfriamiento
adecuados. Además, los desarrolladores tenían que actualizar el sistema
operativo del servidor, defenderlo de ciberataques así como proteger su
integridad física.

_Serverless_ es un modelo de negocio donde un proveedor (por ejemplo, Amazon Web
Services) le da mantenimiento al hardware necesario para desplegar una
aplicación y los consumidores pueden usarlo cuando ellos quieran. Gracias a este
tipo de soluciones, ahora podemos rentar el hardware de AWS y solamente nos
preocupamos por escribir código.

En este sentido, AWS Lambda es el servicio líder del reino _serverless_, pues
nos permite escribir funciones en nuestro lenguaje de
programación preferido y desplegarlo en AWS. Esto significa que no
tenemos que preocuparnos por provisionar o darle mantenimiento a la instancia
en donde nuestra función está hospedada. Lo único que tenemos que hacer es
escribir la función en sí.

## Entrenar un modelo
El proceso de poner un modelo en producción está precedido por la etapa de
entrenamiento del mismo. Dado que el propósito de este artículo es aprender a
produccionalizar un modelo, vamos a pasar rápidamente por el proceso de
entrenamiento. Para esto, descarga [el repositorio](
	https://github.com/ArturoSbr/aws-mlops
), replica el ambiente de Python y ejecuta el script de entrenamiento.

```bash
% cd <path where you downloaded the repo>/aws-mlops # Cambia de directorio
% python3 -m venv my_venv # Crea un nuevo ambiente virtual
% source my_venv/bin/activate # Actívalo
% pip3 install -r requirements.txt # Instala las dependencias
% python3 code/fit-model.py` # Corre el script que exporta un modelo entrenado
```

Estos comandos entrenan un clasificador de potenciación de gradiente usando el
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
Amazon. Desde el punto de vista de un desarrollador, lo único que tenemos que
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
HTTP (HTTP endpoint) para que la clínica pueda invocar al modelo remotamente.
Queremos que los doctores puedan enviar un archivo JSON a este extremo para
activar el código `lambda_function.py`, el cual cargará el modelo (en un objeto
llamado `clf`), recibirá el evento enviado por la clínica, extraerá el cuerpo
(`body`) del mensaje, le pasará los atributos al modelo y regresará una
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
Debemos empacar nuestro código y todas las bibliotecas que utiliza para poder
desplegar nuestra función en una Lambda. La idea es reunir todos los recursos
necesarios para replicar el ambiente de nuestro local en las computadoras de
Amazon.

Como puedes ver, `lambda_function.py` carga el archivo `clf.sav`, el cual es un
objeto de scikit-learn. Lógicamente, esto implica que nuestra función necesitará
cargar scikit-learn, la cual está basada en otras bibliotecas:
```bash
% pip3 show scikit-learn
> Requires: joblib, numpy, scipy, threadpoolctl
```

El mensaje de arriba nos dice que para cargar scikit-learn, tenemos que cargar
joblib, numpy, scipy y threadpoolctl. Por ende, nuestro paquete de despliegue
debe contener:
1. La función (`lambda_function.py`);
2. El modelo serializado (`clf.sav`); y
3. Las bibliotecas guardadas en `my_venv/lib/` (joblib, numpy, threadpoolctl,
   scipy y scikit-learn).

Actualmente, AWS Lambda no permite paquetes mayores a 50 MB y nuestro paquete se
encuentra muy por encima de este corte. Por ende, necesitamos una alternativa
para poder cargar las bibliotecas que nos permitirán cargar el modelo guardado en
el archivo `clf.sav`.

#### Extra - AWS Lambda Layers
**NOTA:** Omite este paso si tu paquete de despliegue cumple con los límites de
carga.

Las famosas [Lambda Layers](
	https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html
) nos permiten almacenar código _fuera_ de nuestras funciones Lambda. Esto
significa que podemos reducir el tamaño de nuestros paquetes de despliegue, pues
nuestras funciones podrán llamar librerías desde las _Layers_ que agreguemos.

Para crear una capa:
1. Ve a [PyPI](https://pypi.org/) y descarga el archivo _wheel_ para _manylinux_
   de la biblioteca que requieras (asegúrate que sea compatible con el _runtime_
   y la arquitectura que elegiste al crear tu función);
2. Desempaca la biblioteca desde tu terminal usando `% wheel unpack
   <your-library.whl>`;
3. Cambia el nombre del folder que se desempacó a _python_;
4. Comprime el folder _python_;
5. Entra a AWS, selecciona tu región y ve a _AWS Lambda_ > _Layers_ >
   _Create a Layer_;
6. Dale un nombre a tu capa y sube el archivo `python.zip` file; y
7. Haz clic en _Create_.

Como ejemplo, crearemos una capa de numpy 1.24.1 para Python 3.9 y procesadores
de 64 bits.
1. Descarga el [archivo wheel](
   numpy-1.24.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
   ) de numpy 1.24.1 para Python 3.9 y procesadores de 64 bits;
2. Desempaca el archivo con `wheel unpack numpy-1.24.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`;
3. Esto creará un folder llamado _numpy-1.24.1_, el cual debes renombrar a
   _python_;
4. Comprime el folder _python_;
5. Accede a AWS y ve a _AWS Lambda_ > _Layers_ > _Create a Layer_;
6. Nombra tu capa y sube el archivo `python.zip`; y
7. Haz clic en _Create_.

Debes repetir este proceso para todas las bibliotecas requeridas por tu función.
Como alternativa, puedes hacer este proceso solamente para las librerías que no
caben en tu paquete de despliegue. Sin embargo, recomiendo hacer esto
individualmente para cada biblioteca individuales, ya que de esta forma podrás
cargarlas por separado en funciones Lambda que crees en el futuro.

Una vez que hayas creado las capas necesarias, puedes proceder a empaquetar los
siguientes archivos:
1. La función (`lambda_function.py`); y
2. El modelo serializado (`clf.sav`).

Para lograrlo, comprime ambos archivos en `deployment-package.zip`.
```bash
cd code/lambda-function/ # Cambia el directorio
zip -r deployment-package.zip . # Comprime todo lo que esté dentro
```

### 3. Crear la función
El siguiente paso consiste en crear la función. Ve a la consola, selecciona la
misma región en donde creaste tus capas y busca _Lambda_. Luego:

1. Ve a _Functions_ > _Create function_ > _Author from scratch_;
2. Configura tu función de la siguiente manera:
   1. Nombra tu función;
   2. Selecciona un runtime de Python 3.9;
   3. Usa una arquitectura de x86_64;
   4. No le hagas cambios a la sección _Permissions_;
   5. Haz clic en
_Advanced Settings_ > _Enable Function URL_ > _Auth type = NONE_; y finalmente
3. Da clic en _Create function_.

![Lambda function configuration](
	https://drive.google.com/uc?export=view&id=1Bu271Wt5XNdu8Fl8rrUc6PtazUgZZckb
)

Estos pasos crearán una función con un extremo abierto. De esta forma,
cualquier persona o aplicación podrá conectarse a nuestra función usando la URL
que generamos al habilitar la opción _Enable Function URL_. En este momento tu
función se debe de ver así:
![Lambda function URL](
	https://drive.google.com/uc?export=view&id=15CnA27KccCbC8HW5vluM5rYi_OCljXp4
)

Recuerda que aún falta incluir scikit-learn y todas sus dependencias en nuestro
paquete de despliegue. Para agregar estas bibliotecas a la función, baja hasta
el final de la página y:
1. Haz clic en _Add a layer_ > _Custom layers_;
2. Selecciona la capa que quieras agregar (joblib, por ejemplo);
3. Selecciona la versión (solamente debería de haber una); y
4. Repite el proceso para las demás bibliotecas (threadpoolctl, numpy, scipy y
   scikit-learn).

Tu función debería verse así:

![Layers added to function](
	https://drive.google.com/uc?export=view&id=1Gioj4T6puagOmGl-SkE9LDRGIrdJELnH
)

## Conéctate a tu función
Recapitulemos lo que hemos hecho. Primero entrenamos y exportamos un modelo de
inteligencia artificial, luego escribimos un archivo de Python que define una
función que recibe un evento y lo mete al modelo para generar una predicción.
Después comprimimos el modelo y la función para crear un paquete de despliegue,
el cual subimos a una Lambda y le agregamos las bibliotecas requeridas para
cargar el modelo usando _Layers_.

Al crear la función habilitamos un extremo HTTP. Nosotros (o cualquier
aplicación) puede invocar la función enviando solicitudes a esta URL. En nuestro
caso de negocio, queremos permitir que los doctores envíen información al
extremo para que el modelo responda con una predicción.

Puntualmente, los doctores enviarán una solicitud POST al extremo de la función.
El evento debe contener todos los atributos requeridos por el modelo en el
cuerpo (`body`) de la solicitud. Como ejemplo, imagina que la prueba de un
paciente resulta en los siguientes valores.

| feature         	| value   |
|---------------------|---------|
| Mean concave points | 0.07951 |
| Worst radius    	| 24.86   |
| Worst texture   	| 26.58   |
| Worst area      	| 1886.0  |
| Mean concave points | 0.01	|

Veremos dos formas en las que un doctor podría enviar esta información al modelo
para generar una predicción.

### Ejemplo 1 - Invoca la función con cURL
El doctor puede enviar la carga desde su terminal de la siguiente manera:
```bash
% curl -X POST \
  	'{your-URL-here}' \
  	-H 'Content-Type: application/json' \
  	-d '{"meanConcavePoints": 0.07951, "worstRadius": 24.86, "worstTexture": 26.58, "worstArea": 1866.0, "worstConcavePoints": 0.01}'
```
Lo que responde con:
```
{"reason":"OK","prediction":1,"status":"200"}
```
Este resultado significa que el modelo detectó señales de cáncer en la prueba
del paciente.

### Ejemplo 2 - Invoca la función desde Python
```python
# Importa una biblioteca para hacer solicitudes a extremos
import requests

# Declara la URL de la función
url = "<your-function's-URL-here>"

# Declara la observación
observation = {
	'meanConcavePoints': 0.07951,
	'worstRadius': 24.86,
	'worstTexture': 26.58,
	'worstArea': 1866.0,
	'worstConcavePoints': 0.01,
}

# Envía la solicitud
req = requests.post(
	url=url,
	json=observation,
)

# Imprime la respuesta
print(req.json())
```
El resultado se ve igual que mediante cURL.
```python
{'reason': 'OK', 'prediction': 1, 'status': '200'}
```

### Ejemplo 3 - Mandar una solicitud mala
Por cautela, veamos qué hace nuestra función para manejar errores. Para
probarlo, enviaremos una observación con solamente un atributo.
```bash
% curl -X POST \
  	'{your-URL-here}' \
  	-H 'Content-Type: application/json' \
  	-d '{"meanConcavePoints": 0.07951}'
```
Lo cual responde con:
```
{"reason":"'body' must contain values for: meanConcavePoints, worstRadius, worstTexture, worstArea, worstConcavePoints","prediction":"","status":"400"}
```

## Conclusión
¡Felicidades! Hemos construido un servicio predictivo completamente _serverless_
usando AWS Lambda. Al seguir este artículo, has aprendido a crear un paquete de
despliegue, agregar bibliotecas externas en forma de capas, habilitar un extremo
HTTP y a activar tu función desde cualquier parte del mundo.

¿Esto es útil? Recuerda que no importa cuánto tiempo pasemos haciendo
validación cruzada sobre nuestras métricas de desempeño, los modelos que
construyamos solo son útiles si otros usuarios o aplicaciones pueden interactuar
con ellos. En este sentido, aprender a produccionalizar tus modelos con AWS
Lambda es una habilidad invaluable que te ayudará a reducir tu _time-to-value_
al permitirte desplegar tus modelos desde la comodidad de tu consola.
