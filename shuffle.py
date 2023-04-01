from random import shuffle

with open('Iris.csv', 'r') as r, open('shuffled_Iris.csv', 'w') as w:
    data = r.readlines()
    header, rows = data[0], data[1:]
    shuffle(rows)
    rows = '\n'.join([row.strip() for row in rows])
    w.write(header + rows)