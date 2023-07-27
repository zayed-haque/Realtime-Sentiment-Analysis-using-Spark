import pickle
import json
import importlib

from pyspark.sql.types import *
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext, Row, SparkSession

from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report, roc_curve
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import IncrementalPCA, LatentDirichletAllocation, PCA, TruncatedSVD

from preprocessing.preprocess import *
from classification_models.logistic_regression import *
from classification_models.multinomial_nb import *
from classification_models.passive_aggressive import *
from clustering_models.kmeans_clustering import *
from clustering_models.birch_clustering import *

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA, TruncatedSVD

import warnings
warnings.filterwarnings("ignore")

TCP_IP = "localhost"
TCP_PORT = 6100

# Create schema
schema = StructType([
	StructField("sentiment", StringType(), False),
	StructField("tweet", StringType(), False),
])

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
									  
# Define MinMax Scaler
minmaxscaler = MinMaxScaler()

# Define CountVectorizer
CountVectorizer.cv_partial_fit = cv_partial_fit
cv = CountVectorizer(lowercase=True, 
					 analyzer = 'word', 
					 stop_words='english', 
					 ngram_range=(1,2))


hv = HashingVectorizer(n_features=2**16, 
					   alternate_sign=False, 
					   lowercase=True, 
					   analyzer = 'word', 
					   stop_words='english', 
					   ngram_range=(1,2))

# Matplotlib
plt.rcParams.update({'figure.figsize':(14, 10), 'figure.dpi':100})
plt.rcParams.update({'font.size': 14})


num_iters = 0

def plot_roc_curve(fper, tper, model):  
	global num_iters
	plt.clf()
	plt.plot(fper, tper, color='orange', label='ROC')
	plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend()

	img_file = open("./roc_curves/" + model + "_" + str(num_iters) + '.eps', "wb+")
	num_iters += 1
	plt.savefig(img_file, format='eps', bbox_inches='tight')


def process(rdd):
	
	global schema, spark, pca, lda, minmaxscaler, cv, hv
	

	records = rdd.collect()
	
	# List of dicts
	dicts = [i for j in records for i in list(json.loads(j).values())]
	
	if len(dicts) == 0:
		return


	df = spark.createDataFrame((Row(**d) for d in dicts), schema)

	df = df_preprocessing(df)
	print('\nAfter cleaning:\n')

	
	tokens_sentiments = df.select('tokens_noStop', 'sentiment').collect()
	
	y_test = np.array([int(row['sentiment']) for row in tokens_sentiments])
	
	tokens = [str(row['tokens_noStop']) for row in tokens_sentiments]
	
	X_test = hv.transform(tokens)


	with open('./trained_models/lr_model_1.pkl', 'rb') as f:
		lr_model_1 = pickle.load(f)
		
	pred_lr_1 = lr_model_1.predict(X_test)
	
	accuracy_lr_1 = np.count_nonzero(np.array(pred_lr_1) == y_test)/y_test.shape[0]	
	print("Accuracy of LR 1:", accuracy_lr_1)
	with open('./test_eval_metrics/lr_1.txt', "a") as ma:
		ma.write(str(accuracy_lr_1)+'\n')

	fper, tper, _ = roc_curve(y_test, pred_lr_1, pos_label=4) 
	plot_roc_curve(fper, tper, 'lr_model_1')

	# Model 2
	with open('./trained_models/lr_model_2.pkl', 'rb') as f:
		lr_model_2 = pickle.load(f)
		
	pred_lr_2 = lr_model_2.predict(X_test)
	
	accuracy_lr_2 = np.count_nonzero(np.array(pred_lr_2) == y_test)/y_test.shape[0]	
	print("Accuracy of LR 2:", accuracy_lr_2)
	with open('./test_eval_metrics/lr_2.txt', "a") as ma:
		ma.write(str(accuracy_lr_2)+'\n')

	fper, tper, _ = roc_curve(y_test, pred_lr_2, pos_label=4) 
	plot_roc_curve(fper, tper, 'lr_model_2')

	# Model 3
	with open('./trained_models/lr_model_3.pkl', 'rb') as f:
		lr_model_3 = pickle.load(f)
		
	pred_lr_3 = lr_model_3.predict(X_test)
	
	accuracy_lr_3 = np.count_nonzero(np.array(pred_lr_3) == y_test)/y_test.shape[0]	
	print("Accuracy of LR 3:", accuracy_lr_3)
	with open('./test_eval_metrics/lr_3.txt', "a") as ma:
		ma.write(str(accuracy_lr_3)+'\n')

	fper, tper, _ = roc_curve(y_test, pred_lr_3, pos_label=4) 
	plot_roc_curve(fper, tper, 'lr_model_3')

	with open('./trained_models/multi_nb_model_1.pkl', 'rb') as f:
		multi_nb_model_1 = pickle.load(f)
		
	pred_mnb_1 = multi_nb_model_1.predict(X_test)	
	accuracy_mnb_1 = np.count_nonzero(np.array(pred_mnb_1) == y_test)/y_test.shape[0]	
	print("Accuracy of NB 1:", accuracy_mnb_1)
	
	with open('./test_eval_metrics/mnb_1.txt', "a") as ma:
		ma.write(str(accuracy_mnb_1)+'\n')
	
	fper, tper, _ = roc_curve(y_test, pred_mnb_1, pos_label=4) 
	plot_roc_curve(fper, tper, 'multi_nb_model_1')
	
	# Model 2
	with open('./trained_models/multi_nb_model_2.pkl', 'rb') as f:
		multi_nb_model_2 = pickle.load(f)
		
	pred_mnb_2 = multi_nb_model_2.predict(X_test)	
	accuracy_mnb_2 = np.count_nonzero(np.array(pred_mnb_2) == y_test)/y_test.shape[0]	
	print("Accuracy of NB 2:", accuracy_mnb_2)
	
	with open('./test_eval_metrics/mnb_2.txt', "a") as ma:
		ma.write(str(accuracy_mnb_2)+'\n')

	fper, tper, _ = roc_curve(y_test, pred_mnb_2, pos_label=4) 
	plot_roc_curve(fper, tper, 'multi_nb_model_2')
	
	# Model 3
	with open('./trained_models/multi_nb_model_3.pkl', 'rb') as f:
		multi_nb_model_3 = pickle.load(f)
		
	pred_mnb_3 = multi_nb_model_3.predict(X_test)	
	accuracy_mnb_3 = np.count_nonzero(np.array(pred_mnb_3) == y_test)/y_test.shape[0]	
	print("Accuracy of NB 3:", accuracy_mnb_3)
	
	with open('./test_eval_metrics/mnb_3.txt', "a") as ma:
		ma.write(str(accuracy_mnb_3)+'\n')

	fper, tper, _ = roc_curve(y_test, pred_mnb_3, pos_label=4) 
	plot_roc_curve(fper, tper, 'multi_nb_model_3')

	with open('./trained_models/pac_model_1.pkl', 'rb') as f:
		pac_model_1 = pickle.load(f)
		
	pred_pac_1 = pac_model_1.predict(X_test)	
	accuracy_pac_1 = np.count_nonzero(np.array(pred_pac_1) == y_test) / y_test.shape[0]	
	print("Accuracy of PAC 1:", accuracy_pac_1)
	
	with open('./test_eval_metrics/pac_1.txt', "a") as ma:
		ma.write(str(accuracy_pac_1)+'\n')
		
	fper, tper, _ = roc_curve(y_test, pred_mnb_1, pos_label=4) 
	plot_roc_curve(fper, tper, 'pac_model_1')
	
	# Model 2
	with open('./trained_models/pac_model_2.pkl', 'rb') as f:
		pac_model_2 = pickle.load(f)
		
	pred_pac_2 = pac_model_2.predict(X_test)	
	accuracy_pac_2 = np.count_nonzero(np.array(pred_pac_2) == y_test) / y_test.shape[0]	
	print("Accuracy of PAC 2:", accuracy_pac_2)
	
	with open('./test_eval_metrics/pac_2.txt', "a") as ma:
		ma.write(str(accuracy_pac_2)+'\n')
		
	fper, tper, _ = roc_curve(y_test, pred_mnb_2, pos_label=4) 
	plot_roc_curve(fper, tper, 'pac_model_2')
	
	# Model 3
	with open('./trained_models/pac_model_3.pkl', 'rb') as f:
		pac_model_3 = pickle.load(f)
		
	pred_pac_3 = pac_model_3.predict(X_test)	
	accuracy_pac_3 = np.count_nonzero(np.array(pred_pac_3) == y_test) / y_test.shape[0]	
	print("Accuracy of PAC 3:", accuracy_pac_3)
	
	
	with open('./test_eval_metrics/pac_3.txt', "a") as ma:
		ma.write(str(accuracy_pac_3)+'\n')
		
	fper, tper, _ = roc_curve(y_test, pred_mnb_3, pos_label=4) 
	plot_roc_curve(fper, tper, 'pac_model_3')

	svd = TruncatedSVD(n_components=2)
	
	with open('./test_kmeans_iteration', "r") as ni:
		k_num_iters = int(ni.read())
	k_num_iters += 1
	with open('./test_kmeans_iteration', "w") as ni:
		ni.write(str(k_num_iters))
			
	with open('./trained_models/kmeans_model.pkl', 'rb') as f:
		kmeans_model = pickle.load(f)

	predictions_kmeans = kmeans_model.predict(X_test)	
	
	preds_1_kmeans = [4 if i == 1 else 0 for i in predictions_kmeans]
	preds_2_kmeans = [0 if i == 1 else 4 for i in predictions_kmeans]
	
	accuracy_1_kmeans = np.count_nonzero(np.array(preds_1_kmeans) == np.array(y_test)) / y_test.shape[0]
	accuracy_2_kmeans = np.count_nonzero(np.array(preds_2_kmeans) == np.array(y_test)) / y_test.shape[0]
			
	print('Accuracy of KMeans: ', accuracy_1_kmeans, accuracy_2_kmeans)
	
	with open('./test_eval_metrics/kmeans.txt', "a") as ma:
		ip=str(accuracy_1_kmeans)+","+str(accuracy_2_kmeans)+'\n'
		ma.write(ip)
	
	X_test_plot = svd.fit_transform(X_test)

	figure, axis = plt.subplots(1, 2)
	axis[0].scatter(X_test_plot[:, 0], X_test_plot[:, 1], c=predictions_kmeans)
	axis[0].set_title('KMeans Clusters')
	axis[0].set_xlabel('Feature 1')
	axis[0].set_ylabel('Feature 2')
	
	axis[1].scatter(X_test_plot[:, 0], X_test_plot[:, 1], c=y_test)
	axis[1].set_title('Original Labels')
	axis[1].set_xlabel('Feature 1')
	axis[1].set_ylabel('Feature 2')
	
	if not os.path.isdir('./test_cluster_plots'):
		os.mkdir('./test_cluster_plots')

	img_file = open("./test_cluster_plots/KMeans_Batch_" + str(k_num_iters) + '.eps', "wb+")
	plt.savefig(img_file, format='eps', bbox_inches='tight')
	plt.clf()

if __name__ == '__main__':


	if not os.path.isdir('./test_eval_metrics'):
		os.mkdir('./test_eval_metrics')
	
	if not os.path.isdir('./roc_curves'):
		os.mkdir('./roc_curves')
		
	with open('./test_kmeans_iteration', "w") as ni:
		ni.write('0')

	lines = ssc.socketTextStream(TCP_IP, TCP_PORT)

	json_str = lines.flatMap(lambda x: x.split('\n'))
	
	lines.foreachRDD(process)


	ssc.start()			 
	ssc.awaitTermination()  



