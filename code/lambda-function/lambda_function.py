# Imports
from pickle import load

# Declare expected keys
expected_keys = [
    'Mean concave points',
    'Worst radius',
    'Worst texture',
    'Worst area',
    'Worst concave points'
]

# Load model
clf = load(
    file=open('./results/clf.sav', 'rb')
)

# Define function
def lambda_handler(event, context):

    # A valid event is passed
    try:
        
        # Check if event is a dict with correct keys
        assert(
            isinstance(event, dict)
            and list(event.keys()) == expected_keys
        )

        # Load request
        x = list(event.values())

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
        err = f'Event must have the following keys:\n{expected_keys}'

        # Return
        return {
            'status': '400',
            'reason': err,
            'prediction': ''
        }
