# Linear Regression Model for Housing Prices
# Data set : Personal NumPy Data Set

# Importing necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Sample data (My data on 500 Houses)
# Assuming x represents features (square footage, bedrooms, bathrooms)
# and y represents the target variable (house prices)
x = np.array([[272,2,1], [147,1,1], [217,1,1], [292,1,1], [351,1,1], [295,2,1], [109,2,1], [311,2,1], [377,2,1], [342,2,1], 
              [392,1,1], [187,2,1], [170,1,1], [188,1,1], [293,2,1], [139,2,1], [187,2,1], [274,2,1], [188,2,1], [265,1,1], 
              [125,1,1], [172,2,1], [365,1,1], [215,1,1], [343,2,1], [297,2,1], [199,2,1], [277,1,1], [343,1,1], [385,2,1], 
              [247,2,1], [247,1,1], [388,1,1], [365,1,1], [285,2,1], [227,2,1], [132,1,1], [131,2,1], [302,1,1], [344,2,1], 
              [251,1,1], [263,2,1], [283,2,1], [128,2,1], [390,2,1], [228,1,1], [228,1,1], [153,1,1], [138,2,1], [344,2,1], 
              [373,2,1], [205,2,1], [142,1,1], [131,1,1], [357,1,1], [157,1,1], [391,1,1], [219,1,1], [367,2,1], [182,2,1], 
              [191,1,1], [199,2,1], [153,1,1], [221,1,1], [184,2,1], [303,1,1], [362,2,1], [147,1,1], [227,1,1], [231,1,1], 
              [280,1,1], [243,2,1], [248,1,1], [327,1,1], [379,2,1], [307,2,1], [148,1,1], [169,1,1], [269,2,1], [263,1,1], 
              [195,2,1], [297,1,1], [194,2,1], [356,1,1], [278,2,1], [392,2,1], [198,2,1], [142,1,1], [301,2,1], [100,1,1], 
              [143,2,1], [123,2,1], [287,1,1], [230,1,1], [198,1,1], [162,2,1], [322,2,1], [223,1,1], [182,1,1], [327,1,1], 
              [248,1,1], [309,1,1], [150,1,1], [370,2,1], [141,1,1], [158,1,1], [293,1,1], [136,1,1], [366,2,1], [186,1,1], 
              [143,1,1], [111,1,1], [358,2,1], [180,2,1], [132,1,1], [282,2,1], [228,1,1], [394,1,1], [375,1,1], [274,2,1], 
              [142,1,1], [284,1,1], [177,1,1], [386,2,1], [380,2,1], [225,2,1], [358,1,1], [103,1,1], [194,2,1], [326,1,1], 
              [369,2,1], [396,2,1], [119,1,1], [195,1,1], [348,1,1], [280,1,1], [370,1,1], [360,1,1], [337,1,1], [239,1,1], 
              [186,2,1], [209,1,1], [284,2,1], [116,1,1], [252,2,1], [249,1,1], [210,1,1], [125,1,1], [217,2,1], [183,1,1], 
              [261,2,1], [328,2,1], [351,1,1], [221,1,1], [387,1,1], [113,2,1], [284,1,1], [252,2,1], [179,2,1], [141,2,1], 
              [374,1,1], [140,2,1], [307,1,1], [367,1,1], [266,1,1], [211,1,1], [229,1,1], [323,2,1], [316,1,1], [124,1,1], 
              [167,1,1], [359,2,1], [334,2,1], [304,1,1], [391,2,1], [314,1,1], [289,2,1], [297,1,1], [315,2,1], [143,2,1], 
              [132,1,1], [111,2,1], [204,1,1], [312,2,1], [238,1,1], [282,1,1], [225,2,1], [256,1,1], [211,1,1], [358,1,1], 
              [127,2,1], [317,1,1], [251,2,1], [274,1,1], [248,1,1], [129,1,1], [167,2,1], [135,1,1], [395,1,1], [173,2,1], 
              [397,2,1], [318,2,1], [359,1,1], [387,2,1], [365,1,1], [127,2,1], [299,2,1], [161,2,1], [144,2,1], [390,2,1], 
              [188,1,1], [133,1,1], [233,2,1], [332,1,1], [355,2,1], [136,1,1], [356,1,1], [390,2,1], [297,2,1], [354,1,1], 
              [180,1,1], [236,1,1], [289,2,1], [229,2,1], [309,1,1], [391,1,1], [268,1,1], [392,1,1], [276,1,1], [125,1,1], 
              [391,2,1], [214,2,1], [386,2,1], [129,2,1], [341,2,1], [389,2,1], [246,2,1], [373,2,1], [321,1,1], [102,1,1], 
              [169,1,1], [144,1,1], [353,1,1], [211,1,1], [191,1,1], [139,2,1], [250,1,1], [245,2,1], [298,1,1], [374,2,1], 
              [143,1,1], [183,2,1], [397,2,1], [193,1,1], [274,2,1], [301,2,1], [128,1,1], [309,2,1], [205,1,1], [163,1,1], 
              [116,1,1], [206,2,1], [264,2,1], [194,1,1], [124,1,1], [216,2,1], [291,1,1], [295,1,1], [236,2,1], [193,1,1], 
              [338,2,1], [187,2,1], [260,1,1], [247,1,1], [172,2,1], [187,2,1], [113,2,1], [181,1,1], [220,1,1], [303,2,1], 
              [320,1,1], [381,1,1], [388,2,1], [370,2,1], [384,1,1], [376,1,1], [122,1,1], [327,1,1], [183,2,1], [235,2,1], 
              [161,2,1], [241,2,1], [105,1,1], [356,1,1], [236,2,1], [307,1,1], [239,1,1], [104,2,1], [382,1,1], [174,1,1], 
              [319,1,1], [327,2,1], [374,1,1], [390,2,1], [151,2,1], [258,1,1], [374,2,1], [153,2,1], [194,1,1], [159,1,1], 
              [254,1,1], [224,1,1], [263,1,1], [158,2,1], [206,2,1], [301,2,1], [369,1,1], [138,2,1], [267,2,1], [236,1,1], 
              [113,2,1], [308,1,1], [122,1,1], [364,1,1], [327,1,1], [362,2,1], [309,2,1], [160,2,1], [156,2,1], [337,2,1], 
              [251,1,1], [380,2,1], [104,1,1], [200,2,1], [130,2,1], [154,2,1], [253,2,1], [376,2,1], [285,1,1], [129,1,1], 
              [219,2,1], [107,2,1], [205,1,1], [237,1,1], [282,2,1], [309,2,1], [248,2,1], [359,2,1], [226,1,1], [398,1,1], 
              [165,1,1], [120,1,1], [392,1,1], [212,1,1], [238,2,1], [337,2,1], [143,1,1], [163,1,1], [365,1,1], [191,1,1], 
              [211,2,1], [374,2,1], [326,1,1], [325,1,1], [231,2,1], [328,1,1], [228,2,1], [395,2,1], [124,1,1], [286,1,1], 
              [392,2,1], [234,2,1], [103,1,1], [326,2,1], [268,1,1], [384,1,1], [126,2,1], [324,2,1], [348,1,1], [301,1,1], 
              [297,2,1], [235,1,1], [225,1,1], [194,1,1], [172,2,1], [346,2,1], [235,2,1], [295,1,1], [313,2,1], [208,2,1], 
              [202,1,1], [183,1,1], [323,2,1], [356,2,1], [233,2,1], [207,2,1], [107,1,1], [249,2,1], [271,2,1], [146,1,1], 
              [100,1,1], [279,2,1], [138,2,1], [189,2,1], [174,2,1], [326,2,1], [223,1,1], [196,1,1], [382,2,1], [306,1,1], 
              [132,2,1], [215,1,1], [278,2,1], [273,2,1], [333,1,1], [232,2,1], [193,2,1], [294,2,1], [285,2,1], [264,2,1], 
              [290,2,1], [104,1,1], [305,1,1], [326,2,1], [173,2,1], [318,1,1], [126,1,1], [122,1,1], [307,1,1], [190,2,1], 
              [151,2,1], [308,1,1], [161,1,1], [204,1,1], [228,1,1], [194,2,1], [106,1,1], [273,1,1], [345,1,1], [115,2,1], 
              [269,2,1], [266,2,1], [182,1,1], [320,2,1], [269,2,1], [214,1,1], [329,2,1], [319,1,1], [346,2,1], [200,2,1], 
              [321,1,1], [278,1,1], [274,2,1], [193,2,1], [214,1,1], [261,2,1], [368,1,1], [324,1,1], [333,1,1], [166,2,1], 
              [225,2,1], [238,2,1], [212,2,1], [318,2,1], [255,1,1], [284,2,1], [220,1,1], [297,1,1], [390,2,1], [103,1,1], 
              [288,2,1], [338,1,1], [265,1,1], [271,1,1], [311,2,1], [170,2,1], [248,2,1], [384,2,1], [215,1,1], [192,2,1], 
              [202,2,1], [297,1,1], [209,1,1], [200,2,1], [282,1,1], [259,2,1], [181,1,1], [135,2,1], [337,1,1], [343,2,1], 
              [350,1,1], [354,2,1], [381,2,1], [121,2,1], [329,2,1], [253,2,1], [219,2,1], [265,1,1], [298,2,1], [174,1,1]] )

y = np.array([10132, 5480, 8070, 10845, 13028, 10983, 4101, 11575, 14017, 12722,
              14545, 6987, 6331, 6997, 10909, 5211, 6987, 10206, 7024, 9846,
              4666, 6432, 13546, 7996, 12759, 11057, 7431, 10290, 12732, 14313,
              9207, 9180, 14397, 13546, 10613, 8467, 4925, 4915, 11215, 12796,
              9328, 9799, 10539, 4804, 14498, 8477, 8477, 5702, 5174, 12796,
              13869, 7653, 5295, 4888, 13250, 5850, 14508, 8144, 13647, 6802,
              7108, 7431, 5702, 8218, 6876, 11252, 13462, 5480, 8440, 8588,
              10401, 9059, 9217, 12140, 14091, 11427, 5517, 6294, 10021, 9772,
              7283, 11030, 7246, 13213, 10354, 14572, 7394, 5295, 11205, 3741,
              5359, 4619, 10660, 8551, 7367, 6062, 11982, 8292, 6775, 12140,
              9217, 11474, 5591, 13758, 5258, 5887, 10882, 5073, 13610, 6923,
              5332, 4148, 13314, 6728, 4925, 10502, 8477, 14619, 13916, 10206,
              5295, 10549, 6590, 14350, 14128, 8393, 13287, 3852, 7246, 12103,
              13721, 14720, 4444, 7256, 12917, 10401, 13731, 13361, 12510, 8884,
              6950, 7774, 10576, 4333, 9392, 9254, 7811, 4666, 8097, 6812, 9725,
              12204, 13028, 8218, 14360, 4249, 10549, 9392, 6691, 5285, 13879,
              5248, 11400, 13620, 9883, 7848, 8514, 12019, 11733, 4629, 6220,
              13351, 12426, 11289, 14535, 11659, 10761, 11030, 11723, 5359, 4925,
              4175, 7589, 11612, 8847, 10475, 8393, 9513, 7848, 13287, 4767,
              11770, 9355, 10179, 9217, 4814, 6247, 5036, 14656, 6469, 14757,
              11834, 13324, 14387, 13546, 4767, 11131, 6025, 5396, 14498, 6997,
              4962, 8689, 12325, 13203, 5073, 13213, 14498, 11057, 13139, 6701,
              8773, 10761, 8541, 11474, 14508, 9957, 14545, 10253, 4666, 14535,
              7986, 14350, 4841, 12685, 14461, 9170, 13869, 11918, 3815, 6294,
              5369, 13102, 7848, 7108, 5211, 9291, 9133, 11067, 13906, 5332,
              6839, 14757, 7182, 10206, 11205, 4777, 11501, 7626, 6072, 4333,
              7690, 9836, 7219, 4629, 8060, 10808, 10956, 8800, 7182, 12574,
              6987, 9661, 9180, 6432, 6987, 4249, 6738, 8181, 11279, 11881,
              14138, 14424, 13758, 14249, 13953, 4555, 12140, 6839, 8763, 6025,
              8985, 3926, 13213, 8800, 11400, 8884, 3916, 14175, 6479, 11844,
              12167, 13879, 14498, 5655, 9587, 13906, 5729, 7219, 5924, 9439,
              8329, 9772, 5914, 7690, 11205, 13694, 5174, 9947, 8773, 4249,
              11437, 4555, 13509, 12140, 13462, 11501, 5988, 5840, 12537, 9328,
              14128, 3889, 7468, 4878, 5766, 9429, 13980, 10586, 4814, 8171,
              4027, 7626, 8810, 10502, 11501, 9244, 13351, 8403, 14767, 6146,
              4481, 14545, 7885, 8874, 12537, 5332, 6072, 13546, 7108, 7875,
              13906, 12103, 12066, 8615, 12177, 8504, 14683, 4629, 10623, 14572,
              8726, 3852, 12130, 9957, 14249, 4730, 12056, 12917, 11178, 11057,
              8736, 8366, 7219, 6432, 12870, 8763, 10956, 11649, 7764, 7515,
              6812, 12019, 13240, 8689, 7727, 4000, 9281, 10095, 5443, 3741,
              10391, 5174, 7061, 6506, 12130, 8292, 7293, 14202, 11363, 4952,
              7996, 10354, 10169, 12362, 8652, 7209, 10946, 10613, 9836, 10798,
              3889, 11326, 12130, 6469, 11807, 4703, 4555, 11400, 7098, 5655,
              11437, 5998, 7589, 8477, 7246, 3963, 10142, 12806, 4323, 10021,
              9910, 6775, 11908, 10021, 7959, 12241, 11844, 12870, 7468, 11918,
              10327, 10206, 7209, 7959, 9725, 13657, 12029, 12362, 6210, 8393,
              8874, 7912, 11834, 9476, 10576, 8181, 11030, 14498, 3852, 10724,
              12547, 9846, 10068, 11575, 6358, 9244, 14276, 7996, 7172, 7542,
              11030, 7774, 7468, 10475, 9651, 6738, 5063, 12510, 12759, 12991,
              13166, 14165, 545, 12241, 9429, 8171, 9846, 11094,  6479])



# Creating a Model - LinearRegression Model
model = LinearRegression()

# We will divide the entire data into training Data and Testing Data
# we will train the model with 60% of the data
extracted_data = train_test_split( x,y, train_size=0.6, shuffle=False)
x_train, x_test, y_train, y_test = extracted_data

# Now we will train the Model
model.fit( x_train, y_train )

# Our Model is trained with the dataset
# Now we will predict the prices of house with some testing data
y_predict = model.predict( x_test ).astype(np.int32)


print("|Sq.ft | BedRooms | BathRooms | Actual Price | Predicted |")
for i in range(0,len(x_test)//2,2):
  print("|{:^6}|{:^10}|{:^11}|{:^14}|{:^11}|".format(x_test[i][0],x_test[i][1],x_test[i][2],y_test[i],y_predict[i]))

# Calculating Accuracy of the predict Output with the Actual Output
accuracy = accuracy_score(y_test, y_predict)
print(f"\n\nAccuracy of Predicted Output with Actual Output: {accuracy * 100}%")

