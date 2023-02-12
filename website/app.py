from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
import PIL
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.axes_grid1 import ImageGrid
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)


@app.route("/")
def index():
	return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
	input_state = request.form.get("state")
	input_district = request.form.get("district")
	input_season = request.form.get("season")
	input_prev_crop = request.form.get("prev_crop")
	input_image = request.files.get("image")
	input_N = request.form.get("nitrogen_content")
	input_P = request.form.get("phosphorus_content")
	input_K = request.form.get("potassium_content")
	input_pH = request.form.get("pH_value")

	# if input_image:
	#   # Do something with the image, such as saving it to disk
	#   input_image.save("website/test_images/test/"+input_image.filename)
	#   return "Image uploaded"
	# else:
	#   return "no image uploaded"
	final_op = []

	##########CNN MODEL############
	model = pickle.load(open('model_pickles/CNN_model.pickle', 'rb'))

	image_size = 200
	batch_size = 10
	no_of_classes = 8
	class_labels = ['Arid Soil', 'Black Soil', 'Cinder Soil',
					'Laterite Soil', 'Peat Soil', 'Red Soil', 'Saline Soil', 'Yellow Soil']

	target_size = (image_size, image_size)
	input_shape = (image_size, image_size, 3)

	test_datagen = ImageDataGenerator(rescale=1/255)
	test_generator = test_datagen.flow_from_directory('CNN Model/test_images',
													  target_size=(
														  image_size, image_size),
													  batch_size=batch_size,
													  classes=['test'],
													  class_mode='categorical',
													  shuffle=False)

	class_mapping = {v: k for k, v in test_generator.class_indices.items()}

	x, y = next(test_generator)
	y_int = np.argmax(y, axis=-1)

	y_prob = model.predict(test_generator)
	y_pred = y_prob.argmax(axis=-1)
	number_of_examples = len(test_generator.filenames)
	number_of_generator_calls = math.ceil(number_of_examples / (1.0 * batch_size))
	# 1.0 above is to skip integer division

	y_test = []

	for i in range(0, int(number_of_generator_calls)):
		y_test.extend(np.argmax(np.array(test_generator[i][1]), axis=-1))

	df = pd.read_csv('datasets/soil_crop.csv')
	df.set_index('soil_type', inplace=True)

	def find_top_soils(prob):
		op = []
		for i in range(no_of_classes):
			if (prob[i] > 0.05):
				op.append(i)
		return op

	def find_whattogrow(soils):
		op = []
		for i, v in enumerate(soils):
			sub = str(df.at[v, 'what_to_grow'])
			subs = sub.split(",")
			for k in subs:
				op.append(k)
		return list(set(op))

	for i, v in enumerate(y_prob):
		top_idx = find_top_soils(v)
		top = [class_labels[j] for j in top_idx]
		crops_from_soiltype = find_whattogrow(top)





	##########Location and Season###########
	crop_prod = pd.read_csv('datasets/crop_production_v2.csv')
	if input_season == 'Whole Year' or input_season not in set(crop_prod[crop_prod['District_Name'] == input_state]['Season'].unique()):
		crops_from_location = crop_prod.loc[(crop_prod['State_Name'] == input_state) & (
			crop_prod['District_Name'] == input_district), 'Crop'].unique()
	else:
		crops_from_location = crop_prod.loc[(crop_prod['State_Name'] == input_state) & (
			crop_prod['District_Name'] == input_district) & (crop_prod['Season'] == input_season), 'Crop'].unique()




	###########CONDITIONS##############
	defualt_NPK=pd.read_csv('datasets/soil_NPK_values.csv')
	crop_reco=pd.read_csv('datasets/crop_recommendation_v2.csv')
	hum_df=pd.read_csv('datasets/statewise_seasonwise_humidity.csv')
	temp_df=pd.read_csv('datasets/statewise_seasonwise_temperature.csv')
	rain_df=pd.read_csv('datasets/statewise_seasonwise_rainfall.csv')

	input_soil=top[0]
	if input_N==0:
		input_N=defualt_NPK[defualt_NPK['soil_type']==input_soil]['N']
	if input_P==0:
		input_P=defualt_NPK[defualt_NPK['soil_type']==input_soil]['P']
	if input_K==0:
		input_K=defualt_NPK[defualt_NPK['soil_type']==input_soil]['K']
	if input_pH==0:
		input_pH=defualt_NPK[defualt_NPK['soil_type']==input_soil]['pH']

	input_hum=hum_df[hum_df['state']==input_state]['annual_avg']
	input_temp=temp_df[temp_df['state']==input_state]['annual_avg']
	if input_district.lower() in list(rain_df['district'].unique()):
		input_rain=rain_df[(rain_df['state']==input_state) & (rain_df['district']==input_district.lower())]['annual_avg'].mean(skipna=True)
	else:
		input_rain=rain_df[rain_df['state']==input_state]['annual_avg'].mean(skipna=True)

	#building model
	X=crop_reco.drop('label', axis=1)
	y=crop_reco['label']
	scaler = MinMaxScaler()
	scaler.fit(X)
	X = pd.DataFrame(scaler.transform(X), columns=X.columns)
	knn=KNeighborsClassifier(n_neighbors=5)
	knn.fit(X, y)


	#knn = pickle.load(open('model_pickles/KNN_model.pickle'))
	input_X = pd.DataFrame([{'N':input_N, 'P':input_P, 'K':input_K, 'temperature':input_temp, 'humidity':input_hum, 'ph':input_pH, 'rainfall':input_rain}])
	input_X = pd.DataFrame(scaler.transform(input_X), columns=input_X.columns)
	output_y=knn.predict(input_X)
	crops_from_conditions=[output_y[0]]
	indices=knn.kneighbors((input_X),return_distance=False)
	temp=list(crop_reco['label'][j].values for j in indices)
	for i in list(temp[0]):
		crops_from_conditions.append(i)
	crops_from_conditions=list(crops_from_location)
	



	#######Combining Models#######
	#final_op=list(set(crops_from_location).union(set(crops_from_soiltype), set(crops_from_conditions)))
	#choode crop if it appears in atleast 2 of the lists
	print(crops_from_location)
	print(crops_from_soiltype)
	print(crops_from_conditions)
	final_op=list(set(crops_from_location) | set(crops_from_soiltype) & set(crops_from_conditions))
	print("BEFORE REMOVING PREV FINAL OP", final_op)


	#######IF prev crop entered######
	if input_prev_crop in final_op:
		final_op.remove(input_prev_crop)
	# # Initialize the data for the resources N, P, K
	# resources = {'N': [100], 
	#              'P': [100], 
	#              'K': [100]}

	# # Create a dataframe from the resources data
	# df_resources = pd.DataFrame(resources, columns=['N', 'P', 'K'])

	# # Initialize the data for the crops and their required N, P, K values
	# crop_requirement = {'Crop': ['Crop1', 'Crop2', 'Crop3', 'Crop4', 'Crop5'], 
	#                     'N': [80, 40, 20, 10, 5], 
	#                     'P': [60, 40, 30, 20, 10], 
	#                     'K': [50, 30, 20, 10, 5]}

	# # Create a dataframe from the crop requirement data
	# df_crop_requirement = pd.DataFrame(crop_requirement, columns=['Crop', 'N', 'P', 'K'])

	# # Given crop that was planted before
	# prev_crop = 'Crop2'

	# # Find the resources used by the previous crop
	# prev_crop_resources = df_crop_requirement[df_crop_requirement['Crop'] == prev_crop].iloc[:, 1:].values[0]

	# # Calculate the remaining resources
	# remaining_resources = df_resources - prev_crop_resources

	# # Find the crops that use the least of the same resource as the previous crop
	# best_crops = df_crop_requirement[(df_crop_requirement['N'] <= remaining_resources.iloc[0]['N']) & 
	#                                  (df_crop_requirement['P'] <= remaining_resources.iloc[0]['P']) & 
	#                                  (df_crop_requirement['K'] <= remaining_resources.iloc[0]['K'])].sort_values(by=['N', 'P', 'K'])

	# # Get the crops that use the least of the same resources
	# best_crops = best_crops['Crop'].tolist()

	# # Print the best crops
	# print("Best crops to plant: ", best_crops)

	#ranking by profit
	#profit_rank=pickle.load(open('model_pickles/Profit_Rank_model.pickle', 'rb')) pickle giving problems
	profit_rank={'papaya': 1, 'orange': 2, 'banana': 3, 'pomegranate': 4, 'coffee': 5, 'apple': 6, 'cotton': 7, 'watermelon': 8, 'maize': 9, 'mango': 10, 'blackgram': 11, 'lentil': 12, 'coconut': 13, 'tapioca': 14, 'kidneybeans': 15, 'rice': 16, 'wheat': 17, 'chickpea': 18, 'grapes': 19, 'jute': 20, 'mungbean': 21}
	final_op=sorted(final_op, key=lambda x: profit_rank[x], reverse=False)

	soil_desc=pd.read_csv('datasets/soil_description.csv')
	desc_dic={'Arid Soil': 'Arid soil is a type of soil found in areas with low rainfall and high evaporation rates, resulting in limited water availability. It is characterized by low organic matter, high levels of salts and minerals, and a compact, rocky structure.', 
				'Black Soil': 'Black soil, also known as regur soil, is a type of fertile soil that is rich in organic matter and has a dark color due to the presence of iron and aluminum oxides. It is found in regions with high rainfall.', 
				'Cinder Soil': 'Cinder soil is a type of soil that is rich in volcanic ash and cinders and is typically found near volcanic areas. It is well-drained, has good water-holding capacity, and is alkaline in nature.', 
				'Laterite Soil': 'Laterite soil is a type of soil that is found in tropical regions with high rainfall and high temperatures. It is rich in iron and aluminum oxides and is typically red or yellow in color.', 
				'Peat Soil': 'Peat soil is a type of soil that is formed from partially decomposed organic matter, such as plant debris and moss. Peat soil is known for its high organic matter content and acidity.', 
				'Red Soil': 'Red soil is a type of soil that is characterized by its reddish-brown color and is commonly found in tropical and subtropical regions. It is generally fertile, with a good balance of nutrients and water-holding capacity.', 
				'Saline Soil': 'Saline soil is a type of soil that has high levels of salt, making it difficult for most crops to grow. It is commonly found in arid and semi-arid regions and is caused by high evaporation rates, poor drainage, and the accumulation of salts over time.', 
				'Yellow Soil': 'Yellow soil is a type of soil that is characterized by its yellowish color, which is due to the presence of iron and aluminum oxides. It is typically found in tropical and subtropical regions and is known for its good water-holding capacity and fertility.'}
	desc=desc_dic[input_soil]
	print("FINAL OP", final_op)
	result=[input_soil,desc,final_op]
	return render_template('index.html', result=result)



if __name__ == "__main__":
	app.run(debug=True)
