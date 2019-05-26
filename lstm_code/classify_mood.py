import math
def classify(prob):
    if prob >= 0.8:
        mood = 1
    elif prob >= 0.6:
        mood = 2
    elif prob >= 0.4:
        mood = 3
    elif prob >= 0.2:
        mood = 4
    else:
        mood = 5
    return mood

def mood_progress(prob):
    return math.floor(prob*100)