import math

# input a list of anime dicts, output a set of attributes that include genres, studios, source
def build_attribute_set(anime_list: list[dict]):
    attributes = set() # use of set is crucial to easily handle duplicates and use of intersection/union
    for genre in anime_list["genres"]:
        attributes.add(('genre', genre))
    for studio in anime_list["studios"]:
        attributes.add(('studio', studio))
    attributes.add(('source', anime_list["source"])) # only 2 possible source types (anime, manga)
    return attributes

# jaccard similarity: ratio of intersection to union of attribute sets
def jaccard(a_attributes, b_attributes): # attributes of anime a and anime b
    intersection_size = len(a_attributes.intersection(b_attributes))
    union_size = len(a_attributes.union(b_attributes))
    if union_size == 0:
        return 0
    return intersection_size / union_size # higher ratio means higher similarity

# adamic-adar: sum of 1/log(degree) for each common attribute
def adamic_adar(a_attributes, b_attributes):
    common_attributes = a_attributes.intersection(b_attributes) # set of common attributes
    score = 0
    for attr in common_attributes:
        degree_attr = len(attr)  # number of anime that have this attribute
        if degree_attr > 1:
            score += 1 / math.log(degree_attr)
    return score

# preferential attachment: product of degrees (number of attributes) of the two anime
# simply means that the more attributes an anime has, the likelier it is to be linked to another anime, regardless of which attributes they are
def preferential_attachment(a_attributes, b_attributes):
    degree_a = len(a_attributes)
    degree_b = len(b_attributes)
    return degree_a * degree_b

