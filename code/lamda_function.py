# Imports
from pickle import load

# Load model
clf = load(
    file=open('./results/clf.sav', 'rb')
)

# Define function
def lambda_handler(event, context):

    # Load request
    x = list(event.values())

    # Make prediction
    pred = clf.predict([x]).item()

    # Return
    return {
        'status': '200',
        'prediction': pred
    }
