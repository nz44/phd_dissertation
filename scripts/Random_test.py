d = {'A': {'a':5, 'b':6}, 'B': {'c':7, 'd':8}}

e = dict.fromkeys(d.keys())

for k1, v1 in d.items():
    e[k1] = dict.fromkeys(v1)

print(e)