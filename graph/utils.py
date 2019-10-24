import re

def reNum(x):
    if re.search(r'[0-9]', x) is not None and re.search(r'[a-zA-Z]', x) is None:
        return '<num>'
    else:
        return x
