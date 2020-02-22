%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from queencity20.utils.getData import *

from collections import defaultdict
df = getTrainingData()
df.head()

from plotnine import *

df["target"].describe()

for col in df.columns:
    p = ggplot(df , aes(x = "target" , y=col , color=)) + geom_point()
    print(p)
