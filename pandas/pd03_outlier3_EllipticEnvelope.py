import pandas as pd
import numpy as np

aaa = np.array([-10,2,3,4,5,6,700,8,9,10,11,12,50]).reshape(-1,1)
print(aaa.shape)    # (13, 1)

from sklearn.covariance import EllipticEnvelope
outlier = EllipticEnvelope(contamination=.3)
outlier.fit(aaa)
result = outlier.predict(aaa)
print(result)