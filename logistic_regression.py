from reviews import create_data_frame
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt


model = SentenceTransformer('EZlee/e-commerce-bert-base-multilingual-cased')

review_data = create_data_frame('amazon_big')



enc = OneHotEncoder()
min_sample_size = min(review_data['Ratings'].value_counts())

evenly_sampled_reviews = review_data.groupby('Ratings').apply(lambda x: x.sample(min_sample_size)).reset_index(drop=True)

reviews = evenly_sampled_reviews['clean_text']
embeddings = model.encode(reviews)
x = embeddings
y = evenly_sampled_reviews['Ratings']


pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter = 5000,solver = "newton-cg"))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

pipe.fit(x_train, y_train)




predictions = pipe.predict(x_test)
print("Overall test accuracy:",accuracy_score(y_test, predictions))
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=pipe.classes_)
disp.plot()
plt.show()