# Load model
clf = load(
    file=open('../results/clf.sav', 'rb')
)

# Class 0
j0 = """
{
    "Mean concave poinst": 0.1,
    "Worst radius": 1.42,
    "Worst texture": 4.7,
    "Worst area": 14.93,
    "Worst concave points": 0.2
}
"""

# Class 1
j1 = """
{
    "Mean concave poinst": 0.19,
    "Worst radius": 33.13,
    "Worst texture": 23.58,
    "Worst area": 3234.0,
    "Worst concave points": 0.28
}
"""

# Define function
def make_pred(request=None):

    # Load request
    d = loads(request)
    x = [d[k] for k in d]

    # Make prediction
    pred = clf.predict([x]).item()

    # Return
    return {
        'status': '400',
        'prediction': pred
    }