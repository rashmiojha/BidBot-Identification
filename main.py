import sys
import model

def create_model():
    model.create_and_save()

def predict():
    print('Choose an algorithm: ')
    algo = input('1:LR\n2:Random Forest\n-->')
    model.predict_score(algo)

if __name__ == '__main__':
    if (len(sys.argv) == 2):
        if(sys.argv[1] == 'create'):
            create_model()
        elif(sys.argv[1] == 'test'):
            predict()
    else:
        print ('Usage:\tmain.py <Create Model["create"] / Test on full test set["test"]>')
        sys.exit(0)