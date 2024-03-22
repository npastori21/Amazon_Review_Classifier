from reviews import create_data_frame
from sklearn.neural_network import MLPClassifier
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt


model = SentenceTransformer('EZlee/e-commerce-bert-base-multilingual-cased')

review_data = create_data_frame('amazon_big')



min_sample_size = min(review_data['Ratings'].value_counts())
print(min_sample_size)
evenly_sampled_reviews = review_data.groupby('Ratings').apply(lambda x: x.sample(min_sample_size)).reset_index(drop=True)
reviews = evenly_sampled_reviews['clean_text']
embeddings = model.encode(reviews)
x = embeddings
y = evenly_sampled_reviews['Ratings']



pipe = make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(100,100,100,100,100), max_iter = 1500,solver = "adam",learning_rate_init = 1e-3, activation = "relu"))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

pipe.fit(x_train, y_train)
predictions = pipe.predict(x_test)
predictions_train = pipe.predict(x_train)
print("Overall training accuracy:", accuracy_score(y_train,predictions_train))
print("Overall test accuracy:", accuracy_score(y_test,predictions))
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=pipe.classes_)
disp.plot()


plt.show()
