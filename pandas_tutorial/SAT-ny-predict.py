# as intro to pandas
#http://blog.kaggle.com/2013/01/17/getting-started-with-pandas-predicting-sat-scores-for-new-york-city-schools/

import pandas as pd

dir = "/home/prliu/WEEKS/week1/pandas_tut/newyork_tut/"

dsProgReports = pd.read_csv( dir + "School_Progress_Reports_-_All_Schools_-_2009-10.csv")
dsDistrict = pd.read_csv(dir + "2010-2011_Class_Size_-_School-level_detail.csv")
dsAttendEnroll = pd.read_csv( dir + "School_Attendance_and_Enrollment_Statistics_by_District__2010-11_.csv")
dsSATs = pd.read_csv(dir + "SAT__College_Board__2010_School_Level_Results.csv")

dsSATs.info()

# now to see where we're going with this...
# we have 5 files - to which we need to merge together.
# one row per school (looks like the author already desided the unique ID key)
# what are the variables? - dependant/dependant?
#  mean SAT scores are said to be dependant...
#  what is our target variable? (variables?)

# goal is to predict the critical reading meann , mathmatics mean and
#  writing mean for each school. ok.
#
#  we need to dig around the data to see how we're gonna join this data
#  (skiped step)
#  my guesses for this step is related to seperation of schools


# general joining strat:
#
# dsSats join dsClassSize on dsSATs['DBN'] = dsClassSize['SCHOOL CODE']
# join dsProgReports on dsSATs['DBN'] = dsProgReports['DBN']
# join dsDistrict on dsProgReports['DISTRICT'] = dsDistrict['JURISDICTION NAME']
# join dsAttendEnroll on dsProgReports['DISTRICT'] = dsAttendEnroll['District']


pd.DataFrame(data=[dsProgReports['DBN'].take(range(5)), dsSATs['DBN'].take(range(5)), dsClassSize['SCHOOL CODE'].take(range(5))])
# ok I'm stuck and I'm not sure what todo with this...
