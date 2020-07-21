import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn import svm

TEST_SIZE = 0.4

month={
	'Jan': 0,
	'Feb': 1,
	'Mar': 2,
	'May': 4,
	'Apr': 3,
	'June':5,
	'Jul': 6,
	'Aug': 7,
	'Sep': 8,
	'Oct': 9,
	'Nov': 10,
	'Dec': 11
	}


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):


	with open("shopping.csv") as f:
	    reader = csv.reader(f)
	    next(reader)

	    data = []
	    ullu=True
	    for row in reader:
	    	if ullu:
	    		print(row)
	    		ullu=not ullu
	    	evid=row[:17]
	    	if row[-1]=='FALSE':
	    		label=0
	    	else:
	    		label=1
	    	evid[10]=month[evid[10]]
	    	if evid[15]=='Returning_Visitor':
	    		evid[15]=1

	    	else:
	    		evid[15]=0
	    	if evid[16]=='TRUE':
	    		evid[16]=1
	    	else:
	    		evid[16]=0
	    	evid[1]=float(evid[1])
	    	evid[3]=float(evid[3])
	    	evid[5]=float(evid[5])
	    	evid[6]=float(evid[6])
	    	evid[7]=float(evid[7])
	    	evid[8]=float(evid[8])
	    	evid[9]=float(evid[9])
	    	
	    	evid[0]=int(evid[0])
	    	evid[2]=int(evid[2])
	    	evid[10]=int(evid[10])
	    	evid[4]=int(evid[4])
	    	evid[11]=int(evid[11])
	    	evid[12]=int(evid[12])
	    	evid[13]=int(evid[13])
	    	evid[14]=int(evid[14])
	    	evid[15]=int(evid[15])
	    	evid[16]=int(evid[16])
	    	
	    	
	    	
	    	
	    	tp={
	    	'evidence': evid,
	    	'label': label
	    	}

	    	data.append(tp)


	evidence = [row['evidence'] for row in data]
	labels = [row['label'] for row in data]
	return (evidence,labels)


def train_model(evidence, labels):
	model=KNeighborsClassifier(n_neighbors=1)
	#model=GaussianNB()
	model.fit(evidence,labels)
	#print(model)
	return model



def evaluate(labels, predictions):
	speci=0
	sensi=0
	for i in range(len(labels)):
		if labels[i]==1 and predictions[i]==1:
			sensi+=1 
		elif labels[i]==0 and predictions[i]==0:
			speci+=1 
	tot_sensi=sum(labels)
	tot_speci=len(predictions)-sum(labels)
	return (sensi/tot_sensi,speci/tot_speci) 


if __name__ == "__main__":
    main()
