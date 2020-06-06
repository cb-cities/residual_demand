import pandas as pd
import numpy as np
print(pd.__version__) ### 1.04
df = pd.DataFrame({'a':[1]*1682698, 'b':list(range(1682698))})
### works fine
# tmp_df = df['a'].isin([1]*79)
# print(tmp_df.shape)
### Index Error
tmp_df = df['a'].isin([1]*80)
print(tmp_df.shape)

'''
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home1/07427/bingyu/virtualenvs/cities/lib/python3.7/site-packages/pandas/core/series.py", line 4294, in isin
    result = algorithms.isin(self, values)
  File "/home1/07427/bingyu/virtualenvs/cities/lib/python3.7/site-packages/pandas/core/algorithms.py", line 453, in isin
    return f(comps, values)
  File "/opt/apps/intel18/python3/3.7.0/lib/python3.7/site-packages/numpy-1.16.1-py3.7-linux-x86_64.egg/numpy/lib/arraysetops.py", line 594, in in1d
    return ret[rev_idx]
IndexError: index 133088 is out of bounds for axis 0 with size 2
'''