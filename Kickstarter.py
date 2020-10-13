import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score as cvs
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import CatBoostEncoder
from category_encoders import TargetEncoder
from category_encoders import CountEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns


data = pd.read_csv("ks-projects-201801.csv",
                   index_col="ID",
                   parse_dates=["deadline","launched"],
                   infer_datetime_format=True)
var = list(data)
data = data.drop(labels=[1014746686,1245461087, 1384087152, 1480763647, 330942060, 462917959, 69489148]) #Clumsy way to pick off rows. See if there's a smarter way to go about it.
y = [i for i in var if i=="state"]
x = data[var]
x = x.drop(labels=y,axis=1)
y = data[y]

missing = [i for i in x if x[i].isnull().any()]

"""
# Prep for imputing or dropping.

for i in missing:
    if x[i].isnull().mean()< 0.01:
        x = x.dropna(subset=[i],axis=0)
        missing.pop(missing.index(i))
"""

obj_feat = x.select_dtypes(include="object")
dat_feat = x.select_dtypes(include="datetime64[ns]")
dat_feat = dat_feat.assign(dmonth=dat_feat.deadline.dt.month.astype("int64"),
                           dyear = dat_feat.deadline.dt.year.astype("int64"),
                           lmonth=dat_feat.launched.dt.month.astype("int64"),
                           lyear=dat_feat.launched.dt.year.astype("int64"))
dat_feat = dat_feat.drop(labels=["deadline","launched"],axis=1)
num_feat = x.select_dtypes(include=["int64","float64"])

#error = list(dat_feat[dat_feat.lyear==1970].index) Will try to fix more properly later

sns.lineplot(x=dat_feat.dyear,y=data.index)
sns.lineplot(x=dat_feat.lyear,y=data.index)
plt.show()

tx,vx,ty,vy = tts(pd.concat([obj_feat,dat_feat,num_feat],axis=1),
                  y,
                  random_state=0)

#Scaler
sc = StandardScaler()

# Imputation strategies
strat = ["constant","most_frequent","mean","median"]

# Encoding unknowns
oh_unk = ["ignore"]

# Encoder
encoders = [LabelEncoder(),
            OneHotEncoder(handle_unknown=oh_unk[0]),
            TargetEncoder(),
            CatBoostEncoder()]

u = dict(zip(list(obj_feat),[len(obj_feat[i].unique()) for i in obj_feat]))
oh_obj = [i for i in u if u[i]<20]
te_obj = [i for i in u if u[i]>20 and u[i]<25]
cb_obj = [i for i in u if u[i]>100]

# Pipeline time
#Impute and encode
num_imp = Pipeline(steps=[("num_imp",SimpleImputer(strategy=strat[2])),("num_scal",sc)])
obj_imp = Pipeline(steps=[("obj_imp",SimpleImputer(strategy=strat[0]))])
oh_enc = Pipeline(steps=[("oh_enc",encoders[1])])
te_enc = Pipeline(steps=[("te_enc",encoders[2])])
cb_enc = Pipeline(steps=[("cb_enc",encoders[3])])

#Transform
trans = ColumnTransformer(transformers=[(num_imp,tx[list(num_feat)]),
                                        (obj_imp,tx[list(obj_feat)]),
                                        (oh_enc,tx[oh_obj]),
                                        (te_enc,tx[te_obj]),
                                        (cb_enc,tx[cb_obj]),
                                        (sc,tx)])

models = [RandomForestClassifier(random_state=0),
          KNeighborsClassifier(),
          DecisionTreeClassifier(random_state=0)]

model = models[0]

# Chaining it all together
run = Pipeline(steps=[("Transformation",trans),("Model",model)])
run.fit(tx,ty)
pred = run.predict(vx)
