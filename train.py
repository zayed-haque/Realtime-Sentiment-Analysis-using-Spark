import pickle
import json
import importlib

from pyspark.sql.types import *
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext, Row, SparkSession

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import IncrementalPCA, LatentDirichletAllocation

from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import MiniBatchKMeans, Birch

from preprocessing.preprocess import *
from classification_models.logistic_regression import *
from classification_models.multinomial_nb import *
from classification_models.passive_aggressive import *
from clustering_models.kmeans_clustering import *
from clustering_models.birch_clustering import *

import warnings
warnings.filterwarnings("ignore")


TCP_IP = "localhost"
TCP_PORT = 6100

# Create schema
schema = StructType([
	StructField("sentiment", StringType(), False),
	StructField("tweet", StringType(), False),
])


# Create a local StreamingContext with two execution threads
sc = SparkContext("local[2]", "Sentiment")
sc.setLogLevel("WARN")
sql_context = SQLContext(sc)
	
spark = SparkSession \
.builder \
.config(conf=SparkConf()) \
.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


ssc = StreamingContext(sc, 5)


pca = IncrementalPCA(n_components=2)


lda = LatentDirichletAllocation(n_components=5,
								max_iter=1,
								learning_method="online",
								learning_offset=50.0,
								random_state=0)

minmaxscaler = MinMaxScaler()

CountVectorizer.cv_partial_fit = cv_partial_fit
cv = CountVectorizer(lowercase=True, 
					 analyzer = 'word', 
					 stop_words='english', 
					 ngram_range=(1,2))

# Define HashVectorizer 
hv = HashingVectorizer(n_features=2**16, 
					   alternate_sign=False, 
					   lowercase=True, 
					   analyzer = 'word', 
					   stop_words='english', 
					   ngram_range=(1,2))

# Define LR Models

lr_model_1 = SGDClassifier(loss='log')
lr_model_2 = SGDClassifier(loss='hinge')
lr_model_3 = SGDClassifier(loss='perceptron')

# Define NB Models

multi_nb_model_1 = MultinomialNB(alpha=1.0, 
							     class_prior=None, 
							     fit_prior=True)
multi_nb_model_2 = MultinomialNB(alpha=0.5, 
							     class_prior=None, 
							     fit_prior=True)
multi_nb_model_3 = MultinomialNB(alpha=0.7, 
							     class_prior=None, 
							     fit_prior=True)

# Define PA Models

pac_model_1 = PassiveAggressiveClassifier(C = 0.2)
pac_model_2 = PassiveAggressiveClassifier(C = 0.5)
pac_model_3 = PassiveAggressiveClassifier(C = 1.0)

# Define Birch Model
brc_model = Birch(n_clusters=2)

# Define BatchKMeans Model
kmeans_model = MiniBatchKMeans(n_clusters=2, 
							   init='k-means++', 
							   n_init=2, 
							   init_size=1000, 
							   verbose=False, 
							   max_iter=1000)
							   

def process(rdd):
	
	global schema, spark, \
		   pca, lda, minmaxscaler, cv, hv, \
		   lr_model_1, lr_model_2, lr_model_3, \
		   multi_nb_model_1, multi_nb_model_2, multi_nb_model_3, \
		   pac_model_1, pac_model_2, pac_model_3, \
		   kmeans_model, brc_model
	
	records = rdd.collect()
	
	# List of dicts
	dicts = [i for j in records for i in list(json.loads(j).values())]
	
	if len(dicts) == 0:
		return

	# Create a DataFrame with each stream	
	df = spark.createDataFrame((Row(**d) for d in dicts), schema)

	df = df_preprocessing(df)
	print('\nAfter cleaning:\n')

	
	tokens_sentiments = df.select('tokens_noStop', 'sentiment').collect()
	
	sentiments = np.array([int(row['sentiment']) for row in tokens_sentiments])
	
	tokens = [str(row['tokens_noStop']) for row in tokens_sentiments]
	
	sparse_vectors = hv.transform(tokens)

	lr_model_1 = LRLearning(sparse_vectors, sentiments, spark, lr_model_1, 1)
	with open('./trained_models/lr_model_1.pkl', 'wb') as f:
		pickle.dump(lr_model_1, f)
		
	lr_model_2 = LRLearning(sparse_vectors, sentiments, spark, lr_model_2, 2)
	with open('./trained_models/lr_model_2.pkl', 'wb') as f:
		pickle.dump(lr_model_2, f)
		
	lr_model_3 = LRLearning(sparse_vectors, sentiments, spark, lr_model_3, 3)	
	with open('./trained_models/lr_model_3.pkl', 'wb') as f:
		pickle.dump(lr_model_3, f)

	multi_nb_model_1 = MultiNBLearning(sparse_vectors, sentiments, spark, multi_nb_model_1, 1)
	with open('./trained_models/multi_nb_model_1.pkl', 'wb') as f:
		pickle.dump(multi_nb_model_1, f) 
		
	multi_nb_model_2 = MultiNBLearning(sparse_vectors, sentiments, spark, multi_nb_model_2, 2)
	with open('./trained_models/multi_nb_model_2.pkl', 'wb') as f:
		pickle.dump(multi_nb_model_2, f) 

	multi_nb_model_3 = MultiNBLearning(sparse_vectors, sentiments, spark, multi_nb_model_3, 3)
	with open('./trained_models/multi_nb_model_3.pkl', 'wb') as f:
		pickle.dump(multi_nb_model_3, f) 

	pac_model_1 = PALearning(sparse_vectors, sentiments, spark, pac_model_1, 1)
	with open('./trained_models/pac_model_1.pkl', 'wb') as f:
		pickle.dump(pac_model_1, f)

	pac_model_2 = PALearning(sparse_vectors, sentiments, spark, pac_model_2, 2)
	with open('./trained_models/pac_model_2.pkl', 'wb') as f:
		pickle.dump(pac_model_2, f)

	pac_model_3 = PALearning(sparse_vectors, sentiments, spark, pac_model_3, 3)
	with open('./trained_models/pac_model_3.pkl', 'wb') as f:
		pickle.dump(pac_model_3, f)

	with open('./kmeans_iteration', "r") as ni:
		k_num_iters = int(ni.read())
	
	k_num_iters += 1

	kmeans_model = \
			kmeans_clustering(sparse_vectors, sentiments, spark, kmeans_model, k_num_iters)
	
	with open('./kmeans_iteration', "w") as ni:
		ni.write(str(k_num_iters))
		
	with open('./trained_models/kmeans_model.pkl', 'wb') as f:
		pickle.dump(kmeans_model, f)


# Main entry point for all streaming functionality
if __name__ == '__main__':

	if not os.path.isdir('./trained_models'):
		os.mkdir('./trained_models')
		
	if not os.path.isdir('./model_accuracies'):
		os.mkdir('./model_accuracies')
		
	# Create a DStream - represents the stream of data received from TCP source/data server
	# Each record in 'lines' is a line of text
	lines = ssc.socketTextStream(TCP_IP, TCP_PORT)

	json_str = lines.flatMap(lambda x: x.split('\n'))
	
	with open('./birch_iteration', "w") as ni:
		ni.write('0')
	with open('./kmeans_iteration', "w") as ni:
		ni.write('0')
	
	# Process each RDD
	lines.foreachRDD(process)

	# Start processing after all the transformations have been setup
	ssc.start()             # Start the computation
	ssc.awaitTermination()  # Wait for the computation to terminate


