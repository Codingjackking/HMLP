import math

def build_attribute_set(anime_list: list[dict]):
    attributes = set()
    for genre in anime_list["genres"]:
        attributes.add(('genre', genre))
    for studio in anime_list["studios"]:
        attributes.add(('studio', studio))
    attributes.add(('source', anime_list["source"]))
    return attributes

def jaccard(a_attributes, b_attributes):
    intersection_size = len(a_attributes.intersection(b_attributes))
    union_size = len(a_attributes.union(b_attributes))
    if union_size == 0:
        return 0
    return intersection_size / union_size

def adamic_adar(a_attributes, b_attributes):
    common_attributes = a_attributes.intersection(b_attributes)
    score = 0
    for attr in common_attributes:
        degree_attr = len(attr)  # number of anime sharing this attribute
        # idk if this is the right way to calculate degree_attr, but it should be the number of anime that have this attribute
        if degree_attr > 1:
            score += 1 / math.log(degree_attr)
    return score