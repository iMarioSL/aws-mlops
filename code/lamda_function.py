# Imports
from pickle import load

# Load model
clf = load(
    file=open('./results/clf.sav', 'rb')
)

# Define function
def lambda_handler(event, context):

    # Load request
    d = event
    x = list(d.values())

    # Make prediction
    pred = clf.predict([x]).item()

    # Return
    return {
        'status': '400',
        'prediction': pred
    }
