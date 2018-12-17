from __future__ import print_function
from rdkit import Chem

## Read the file
supplier = Chem.SDMolSupplier('data/cas_4337.sdf')
len(supplier)

import numpy as np
from rdkit.Chem import AllChem

info = {} # will be mutated in the next function
## calculate the Morgan Fingerprints for every molecule in the supplier
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048, bitInfo=info) for mol in supplier]
## convert it from bit vector to NumPy array
fingerprints = np.array(fingerprints)

## KAREEM: these molecules I got from Kristina!
val_ids = [6,   10,   29,   32,   42,   58,   72,   83,   98,  100,  128, 
        145,  148,  168,  171,  205,  208,  237,  244,  285,  290,  291,
         300,  312,  332,  334,  335,  347,  356,  369,  371,  377,  407,
         424,  456,  458,  470,  472,  486,  514,  515,  528,  557,  563,
         599,  610,  616,  628,  640,  701,  704,  722,  764,  794,  818,
         821,  840,  850,  856,  859,  874,  878,  882,  898,  901,  925,
         936,  945,  957,  974,  977, 1013, 1019, 1030, 1038, 1047, 1049,
        1072, 1073, 1100, 1159, 1168, 1187, 1190, 1194, 1201, 1202, 1233,
        1247, 1258, 1264, 1273, 1283, 1288, 1300, 1302, 1319, 1339, 1349,
        1402, 1413, 1416, 1422, 1426, 1435, 1454, 1465, 1483, 1502, 1513,
        1515, 1520, 1548, 1576, 1604, 1606, 1621, 1650, 1695, 1696, 1711,
        1714, 1716, 1725, 1743, 1746, 1752, 1780, 1788, 1794, 1799, 1813,
        1826, 1866, 1886, 1901, 1903, 1921, 1929, 1940, 1969, 1970, 1997,
        1998, 2008, 2010, 2011, 2018, 2023, 2046, 2060, 2064, 2080, 2081,
        2131, 2171, 2182, 2203, 2212, 2224, 2231, 2241, 2246, 2283, 2294,
        2295, 2297, 2327, 2329, 2331, 2349, 2357, 2360, 2365, 2397, 2413,
        2417, 2418, 2421, 2448, 2467, 2510, 2516, 2528, 2533, 2549, 2562,
        2601, 2604, 2606, 2609, 2611, 2632, 2644, 2653, 2677, 2682, 2685,
        2692, 2703, 2708, 2714, 2719, 2726, 2732, 2759, 2761, 2776, 2780,
        2817, 2818, 2829, 2837, 2857, 2858, 2884, 2899, 2902, 2905, 2911,
        2939, 2975, 2977, 2986, 3007, 3009, 3018, 3024, 3038, 3066, 3087,
        3098, 3107, 3117, 3122, 3139, 3157, 3161, 3164, 3217, 3223, 3233,
        3263, 3265, 3271, 3290, 3295, 3307, 3313, 3317, 3321, 3382, 3384,
        3388, 3400, 3409, 3412, 3419, 3423, 3449, 3470, 3487, 3488, 3503,
        3509, 3511, 3539, 3562, 3626, 3637, 3654, 3662, 3663, 3668, 3671,
        3688, 3689, 3695, 3710, 3726, 3743, 3744, 3782, 3791, 3794, 3808,
        3809, 3841, 3849, 3874, 3910, 3912, 3925, 3945, 3950, 3958, 3959,
        3962, 3964, 3967, 3978, 3993, 4009, 4010, 4055, 4057, 4085, 4089,
        4096, 4099, 4107, 4112, 4129, 4135, 4151, 4155, 4196, 4209, 4216,
        4234, 4236, 4251, 4267, 4283, 4317, 4326, 4335

]

train_samples = []
valid_samples = []
for i in range(len(fingerprints)):
    if i in val_ids:
        valid_samples.append(fingerprints[i])
    else:
        train_samples.append(fingerprints[i])
train_samples = np.array(train_samples)
valid_samples = np.array(valid_samples)

targets = []
for mol in supplier:
    if mol.GetProp("Ames test categorisation") == "mutagen":
        targets.append(1)
    else:
        targets.append(0)

train_labels = []
valid_labels = []
for i in range(len(targets)):
    if i in val_ids:
        valid_labels.append(targets[i])
    else:
        train_labels.append(targets[i])

train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)

from sklearn.preprocessing import StandardScaler
#Scale fingerprints to unit variance and zero mean
st = StandardScaler()
train_samples = st.fit_transform(train_samples)
valid_samples = st.transform(valid_samples)


import os

# importing all libraries that we'd need
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy

from sklearn import cross_validation
from sklearn.metrics import roc_auc_score # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores

from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm


model = Sequential()
model.add(Dense(256, input_dim=train_samples.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=int(256/8), activation='selu'))
model.add(Dense(output_dim=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])
model.summary()

for epoch in range(100):
    model.fit(train_samples, train_labels, batch_size=32, epochs=1)
    predictions = model.predict(valid_samples)
    auc = roc_auc_score(valid_labels, predictions)
    print(auc)
