# Calculate document distance given two files
# Uses cosine formula described on Wikipedia: https://en.wikipedia.org/wiki/Cosine_similarity
# Based on implementation by vpekar: http://stackoverflow.com/q/15173225

import math
import re
from collections import Counter

# regular expression
# \w matches any alphanumeric character and the underscore
# + causes the RE to match 1 or more repetitions of the preceding RE
WORD = re.compile(r'\w+')


def TEXTToVector(TEXT):
    words = WORD.findall(TEXT)
    # unordered collection where elements are stored as dict keys, and counts are stored as dict vals
    return Counter(words)


def cosDistance(vector1, vector2):
    # set of unordered collection of unique items
    intersection = set(vector1.keys()) & set(vector2.keys())  # return set with elements in intersection
    numerator = sum([vector1[x] * vector2[x] for x in intersection])

    sum1 = sum([vector1[x] ** 2 for x in vector1.keys()])
    sum2 = sum([vector2[x] ** 2 for x in vector2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def readFile(fileName):
    #return open("../data/" + fileName, 'r').read()
    return open(fileName, 'r').read()


TEXT1 = readFile("nowisthe.txt")
TEXT2 = readFile("thequick.txt")

vector1 = TEXTToVector(TEXT1)
vector2 = TEXTToVector(TEXT2)

cosine = cosDistance(vector1, vector2)

print("Cosine Distance:\t", cosine)
print("percentange of simlarity:\t",cosine*100,"%")
