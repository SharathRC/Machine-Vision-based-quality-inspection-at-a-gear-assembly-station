

from collections import namedtuple
import pandas as pd


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

examples = pd.read_csv('test_labels.csv')
grouped = split(examples, 'filename')
for group in grouped:
    class_id = group.filename.split("_")[0]
    print(filename, class_id)
    print(type(filename))
    break