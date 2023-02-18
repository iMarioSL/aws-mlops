# Imports
from json import loads
from pickle import load

# Declare expected keys
expected_keys = [
    'meanConcavePoints',
    'worstRadius',
    'worstTexture',
    'worstArea',
    'worstConcavePoints'
]

# Load model
clf = load(
    file=open('./clf.sav', 'rb')
)

# Define function
def lambda_handler(event, context):

    # A valid event.body is passed
    try:

        # Extract body and turn to dict
        body = loads(event['body'])
        
        # Check if event is a dict with correct keys
        assert(
            list(body.keys()) == expected_keys
        )

        # Extract values
        x = list(body.values())

        # Make prediction
        pred = clf.predict([x]).item()

        # Return
        return {
            'status': '200',
            'reason': 'OK',
            'prediction': pred
        }

    # Invalid event is passed:
    except:

        # Error message
        err = f"'body' must contain values for: {', '.join(expected_keys)}"

        # Return
        return {
            'status': '400',
            'reason': err,
            'prediction': ''
        }
