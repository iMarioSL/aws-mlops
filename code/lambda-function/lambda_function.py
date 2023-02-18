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
        req = loads(event['body'])
        
        # Check if event is a dict with correct keys
        assert(
            list(event.keys()) == expected_keys
        )

        # Extract values
        x = list(req.values())

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
        err = f"Body must contain values for: {', '.join(expected_keys)}"

        # Return
        return {
            'status': '400',
            'reason': err,
            'prediction': ''
        }
