# 10 minutes of pandas example

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# create a series
s = pd.Series([1,3,5, np.nan, 6, 8])

dates = pd.date_range('20130101', periods=6)
dates
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list("ABCD"))
df

df.index

df.columns

df.values

#not sure what kind of stats this is giving me though...
df.describe
# transpose.
df.T

# sort by axis
df.sort_index(axis=1, ascending=False)

# sort by value
df.sort_values(by='B')
# pass in a dict of objects
# to form a dataframe

# this doesn't work...
#pd.DataFrame(data=[dsProgReports['DBN'].take(range(5)), dsSATs['DBN'].take(range(5)), dsClassSize['SCHOOL CODE'].take(range(5))])

# Selection - you'll want to use .at , .iat , .loc, .iloc, .ix for production CODE
#

df['A']
#slice
df[0:3]

#select by label
df.loc[dates[0]]
# select multi-axis
df.loc[:,['A', 'B']]

# selection by position
df.iloc[3]
df.iloc[3:5, 0:2] # integer slices  iloc[row, col]

df.iloc[[1,2,4],[0,2]] #by list of integer positions

#Boolean indexing

df[df.A > 0]
# overall so far:
# - requires context of the data
# - some understanding of the questions toward the data (and everything that goes with that )
