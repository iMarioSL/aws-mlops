# Productionizing a Machine Learning Model with AWS Lambda

In this article we will train a simple gradient boosting ensemble of decision
trees to predict if digitized images of cells are cancerous or not. We will then
export this model and enable it for online consumption using AWS Lambda.

The end result will be a fully serverless prediction service that users can
interact with to preidct if a given image is cancerous or not.

## What do we mean by _productionizing a model_?
According to Stephen Hawking, a _good model_ is an elegant simplification of
reality that intakes characteristics to accurately predict an outcome. Now, no
matter how accurate and parsimonious a model may be, it is only useful
insofar as other people can actually _use_ it to make predictions.

A lot of us scientists are able to apply complex Machine Learning algorithms to
learn patterns from data and accurately predict the future. However, most of us
do not know how to make our models available to the rest of the world. As a
consequence, a lot of outstanding Machine Learning projects never make it out of
their authors' hard drives and die as a noble effort to be useful.

Putting a model in production (i.e., _productioninzing a model_) is the process
of making it available to users _outside_ of your local environment. In other
words, _productionizing a model_ refers to allowing users who cannot physically
interact with your computer to use your model in the same way you would.

## What do we mean by _serverless_?
In order to productionize a model back in the day, developers had to go to their
closest hardware store to buy a server, install it in their garage, host their
application in it and connect it to the internet so that their model could be
used by people all over the world. As you can imagine, this is a highly
expensive process, as it involves purchasing a server with the appropriate
amount of RAM, storage space, network card, cooling system, etc. On top of that,
developers had to worry about patching the server's operating system, updating
the dependencies used by the model, setting up firewalls to fend off hackers and
keeping an eye on the neighborhood kids to make sure that no one would tamper
with the server itself!

_Serverless_ is a business model where a vendor (i.e., Amazon Web Services)
provides and maintains the hardware needed to host an application and consumers
can use it on-demand to deploy software. This means we can simply rent
infrastructure from AWS and forget about buying a server, maintaining its
physical integrity, watching a 10-hour crash course about HTTPS, etc.

## Why AWS Lambda?
AWS Lambda is the _crème de la crème_ of the serverless kingdom. It is a service
that allows us to write functions in our prefered programming language and
deploy them on a machine owned and maintained by AWS. This means we do not have
to worry about provisioning or maintaining the instance that hosts the function.
All we need to do is write the function itself!

## Training a model
