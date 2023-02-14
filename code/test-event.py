from json import dumps
from lambda_function import lambda_handler

e = {
  #"Pepe": "Frog",
  "Mean concave points": 0.1,
  "Worst radius": 1.42,
  "Worst texture": 4.7,
  "Worst area": 14.93,
  "Worst concave points": 0.2
}

print(
    dumps(
        lambda_handler(event=e, context=None), indent=2
    )
)
