

from scipy.io import arff


data_fpath = "data/dataset_61_iris.arff"

with open(data_fpath, "r") as f:
    data, meta = arff.loadarff(f)


prompt_template = "Q: sepal length is {} cm, sepal width is {} cm, petal length is {} cm, petal width is {} cm,  A: class is {}"

res = []
for row in data:
    res.append(prompt_template.format(row[0], row[1], row[2], row[3], row[4].decode("utf-8")))

with open("prompts/iris_promp.txt", 'w') as f:
    for r in res:
        f.writelines(r+'\n')
