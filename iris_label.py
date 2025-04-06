
labels = []

for i in range(50):
    labels.append(0)

for i in range(50):
    labels.append(1)

for i in range(50):
    labels.append(2)

with open("prompts/iris_promp.label", 'w') as f:
    for val in labels:
       f.write(f"{val}\n")