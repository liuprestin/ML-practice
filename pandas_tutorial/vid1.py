import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

import numpy as np

# https://www.youtube.com/watch?v=0UA49Ds1XXo&list=PLQVvvaa0QuDc-3szzjeP6N6b0aDrrKyL-&index=2

# note: atom with hydrogen installed: shift-enter works!!!
# need to enter each line.

# pandas is like a python dict
#

web_stats = {'day' : [1,2,3], 'visitors' : [43,53,34] ,'bounce_rate': [55, 63 , 666]}

df = pd.DataFrame(web_stats)

print(df)
print(df.head(1))
# choosing what the index is???
#
print(df.set_index("day")) # usig set_index creates new frame

print(df['visitors']) #reference a specific column
print(df.visitors) #be careful with spaces here...

print(df[['bounce_rate','visitors']]) # print more than one column

print(df.visitors.tolist())
print(np.array(df[['bounce_rate','visitors']])) # convert to a numpy array
