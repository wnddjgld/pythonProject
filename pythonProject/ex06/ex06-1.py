import pandas as pd

edges = pd.DataFrame({'source': [0,1,2],
                      'target': [2,2,3],
                      'weight': [3,4,5],
                      'color': ['red', 'blue', 'blue']})
# print(edges)
# print(pd.get_dummies(edges))
# print(pd.get_dummies(edges[["color"]]))

weight_dict = {3:"M", 4:"L", 5:"XL"}
edges["weight_sign"] = edges["weight"].map(weight_dict)
weight_sign = pd.get_dummies(edges["weight_sign"])
print(weight_sign)
print(pd.concat([edges, weight_sign], axis=1))
