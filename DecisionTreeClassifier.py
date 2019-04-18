from math import log
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



class DecisionStump:
   debug = False
   no_data = True
   best_gain = 0
   best_feature = ""
   best_stump = {}
   target_data_list = []
   data = pd.DataFrame()
   keys = []
   num_keys = 0
   target_data_key = []
   target_data = pd.DataFrame()

   def raiseError(self, error):
       raise ValueError(error)

   def __init__(self, data, should_debug = False):
       if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
           self.debug = should_debug
           self.no_data = False
           self.data = data
           self.keys = data.keys()
           self.num_keys = len(self.keys)
           self.target_data_key = self.keys[self.num_keys - 1]
           self.target_data = self.data[self.target_data_key]
           self.calculate_gain_for_features()
       else:
           self.raiseError("Inappropriate data set")


   # calculate entropy for input array [4,0]
   def entropy_cal(self, arr_val):
       total = 0
       for pi_log in arr_val:
           pi_log = pi_log / sum(arr_val)
           if pi_log != 0:
               total += pi_log * log(pi_log, 2)
           else:
               total += 0
       total *= -1
       return total

   # calculate gain for inputs:
   # total_entropy = [4,0] and arr_val = [[4,0],[1,2]...]
   def gain(self, total_entropy, arr_val):
       total = 0
       for subset_val in arr_val:
           total += sum(subset_val) / sum(total_entropy) * self.entropy_cal(subset_val)

       gain = self.entropy_cal(total_entropy) - total
       return gain

   ##
   # Create one dimensional target array value count
   # For playtennis = [9,6]
   ##
   def create_target_array(self, data):
       my_dic = {}
       for i in range(len(data)):
           key = data[i]
           if key in my_dic:
               val = my_dic[key]
               my_dic[key] = val + 1
           else:
               my_dic[key] = 1

       my_list = []
       for j in my_dic:
           my_list.append(my_dic[j])
       return my_list

   ##
   # Create two dimensional feature array value count
   # For outlook = [[9,6], [4,0]....]
   ##
   def create_features_list(self, feature_data, data):
       feature_dic = {}
       for i in range(len(feature_data)):
           key = feature_data[i]
           target = data[i]
           if key in feature_dic:
               target_dic = feature_dic[key]
               if target in target_dic:
                   val = target_dic[target]
                   target_dic[target] = val + 1
               else:
                   target_dic[target] = 1
           else:
               my_dic = {target: 1}
               feature_dic[key] = my_dic
       return feature_dic

   def create_list_of_feature_dic(self, feature_dic):
       list = []
       for i in feature_dic:
           dict = feature_dic[i]
           my_list = []
           for j in dict:
               my_list.append(dict[j])
           list.append(my_list)
       return list

   ##
   # Calculates best gain and feature for the given data set
   ##
   def calculate_gain_for_features(self):
       prev_gain = 0
       prev_feature = ""
       target_data = self.target_data
       target_data_list = self.create_target_array(target_data)
       for i in range(self.num_keys - 1):
           key = self.keys[i]
           feature_dic = self.create_features_list(self.data[key], target_data)
           target_fetaure_list = self.create_list_of_feature_dic(feature_dic)
           gain = self.gain(target_data_list, target_fetaure_list)
           if prev_gain < gain:
               prev_gain = gain
               prev_feature = key

       self.best_gain = prev_gain
       self.best_feature = prev_feature
       if self.debug:
           print("\n__ calculate_gain_for_features __")
           print("Best Feature:", self.best_feature)
           print("Best Gain:", self.best_gain)

   # Create a fit method to calculate the best stump from given training instances
   # Assumptions: Here we are assuming that if the two similar values in entropy array then we consider it 'No'
   # Example:sunny: [2,2] then 'No playtennis
   def fit(self, x_train, y_train):
       if x_train is None or len(x_train) == 0:
           self.raiseError("Inappropriate X training data ")
       elif y_train is None or len(y_train) == 0:
           self.raiseError("Inappropriate Y training data ")
       elif self.no_data:
           self.raiseError("No Model Created Error")

       train = list(x_train[self.best_feature])
       target = list(y_train[list(y_train)[0]])
       dict = self.create_features_list(train, target)
       # print(dict)
       for i in dict:
           my_dic = dict[i]
           prev_outcome = ""
           prev_outcome_count = 0
           for j in my_dic:
               new_outcome_count = my_dic[j]
               if prev_outcome_count < new_outcome_count:
                   prev_outcome = j
                   prev_outcome_count = new_outcome_count
           dict[i] = prev_outcome

       self.best_stump = dict
       if self.debug:
           print("\n__fit __")
           print("Best Stump: ", self.best_stump)

   def predict(self, x_test):
       if x_test is None or len(x_test) == 0:
           self.raiseError("Inappropriate X test data ")
       elif self.no_data:
           self.raiseError("No Model Created Error")

       test = list(x_test[self.best_feature])
       my_predictions = []
       for i in range(len(test)):
           my_predictions.append(self.best_stump[test[i]])

       if self.debug:
               print("\n __ predict __")
               print("Test Set: ", test)
               print("Predictions:", my_predictions)
       return my_predictions


data = pd.read_csv("playtennis.csv")

y_dataset = data.iloc[:, 4]
x_dataset = data.iloc[:, 0:4]

# create the dataframe
x_dataframe = pd.DataFrame(x_dataset)
y_dataframe = pd.DataFrame(y_dataset)

x_train, x_test, y_train, y_test = train_test_split(x_dataframe, y_dataframe, test_size=0.30)
d = DecisionStump(data, False)
d.fit(x_train, y_train)
predictions = d.predict(x_test)
print(predictions)