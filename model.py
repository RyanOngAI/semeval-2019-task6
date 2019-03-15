# Others
import re
import nltk
import itertools
import numpy as np
import pandas as pd
import helper as hp
import tensorflow as tf
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Sklearn library
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Keras library
import keras.backend as K
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Bidirectional, GRU, SpatialDropout1D

class Offensive_Detector():
    
    def __init__(self, m_hyperparameters):
        self.hparams = m_hyperparameters
        if self.hparams['use_pretrained_embedding'] == True:
            self.load_build_embeddings()
    
    def load_build_embeddings(self):
        print("Loading and building embedding matrix")
        embeddings_index = dict()
#        f = open('glove.twitter.27B.200d.txt') 
#        f = open('glove.twitter.27B.100d.txt')
        f = open('glove.42B.300d.txt') # LSTM Console 1
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        embedding_matrix = np.zeros((self.hparams['vocab_size'], 300))
        for word, index in self.hparams['tokenizer'].word_index.items():
            if index > self.hparams['vocab_size'] - 1:
                break
            else:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector    
        
        self.embedding_matrix = embedding_matrix
    
    def f1_macro(self, y_true, y_pred):
        y_pred = K.round(y_pred)
        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    
        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())
    
        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        return K.mean(f1)
        
    def build(self, optimizer = 'adam'):
        
        # Network architecture
        model = Sequential()
        if self.hparams['use_pretrained_embedding'] == True:
            print("\n")
            print("Using pretrained embedding matrix")
            model.add(Embedding(self.hparams['vocab_size'], 300, input_length=50, weights=[self.embedding_matrix], trainable=False)) #trainable = false/true
        else:
            model.add(Embedding(self.hparams['vocab_size'], 100, input_length=50))
        model.add(Bidirectional(LSTM(300, return_sequences = True, dropout=0.35, recurrent_dropout=0.35)))
#        model.add(Bidirectional(GRU(300, return_sequences = True, dropout=0.35, recurrent_dropout=0.35)))
        if self.hparams['convolutional_layer'] == True:
            model.add(Conv1D(128, 4, activation='relu'))
            model.add(MaxPooling1D(pool_size=4))
            #model.add(Flatten())
        if self.hparams['rnn_layer_after_cnn'] == True:
            if self.hparams['rnn_layer'] == 'LSTM':
                model.add(LSTM(100, dropout=0.35, recurrent_dropout=0.35))
            elif self.hparams['rnn_layer'] == 'Bidirectional LSTM':
                model.add(Bidirectional(LSTM(100, dropout=0.35, recurrent_dropout=0.35)))
            elif self.hparams['rnn_layer'] == 'GRU':
                model.add(GRU(100, dropout=0.35, recurrent_dropout=0.35))
            elif self.hparams['rnn_layer'] == 'Bidirectional GRU':
                model.add(Bidirectional(GRU(100, dropout=0.35, recurrent_dropout=0.35)))
#        model.add(SpatialDropout1D(0.5))
        model.add(Dense(128, activation='relu'))
#        model.add(SpatialDropout1D(0.5))
        model.add(Flatten())
#        model.add(Dropout(0.2))
        if self.hparams['num_classes'] == 2:
            model.add(Dense(1, activation='sigmoid'))
        elif self.hparams['num_classes'] > 2:
            model.add(Dense(3, activation='softmax'))
        
        self.model = model
        if self.hparams['num_classes'] == 2:
            if self.hparams['custom_metrics_f1'] == True:
                self.model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy', self.f1_macro])
            else:
                self.model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])            
        elif self.hparams['num_classes'] > 2:
            if self.hparams['custom_metrics_f1'] == True:
                self.model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy', self.f1_macro])
            else:
                self.model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
            
        model.summary()
        return model
        
    def train(self, X_train_val, y_train_val, computed_weights = None, callbacks = None):
        self.model.fit(X_train_val, y_train_val, validation_split = 0.25, epochs = self.hparams['epochs'], class_weight = computed_weights, callbacks=callbacks)
        
    def evaluate(self, X_test, y_test):
        score = self.model.evaluate(X_test, y_test)
        print('Test accuracy:', score[1])
        
        predictions = self.model.predict(X_test)
        predictions_round = [np.round(x) for x in predictions]
        print(classification_report(y_test, np.array(predictions_round)))
        
        def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            """
            
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
        
            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
        
        if self.hparams['num_classes'] == 2:
            cnf_matrix = confusion_matrix(y_test, np.array(predictions_round))
        elif self.hparams['num_classes'] > 2:
            cnf_matrix = confusion_matrix(y_test.argmax(axis = 1), np.array(predictions_round).argmax(axis = 1))
        plot_confusion_matrix(cnf_matrix, classes = model_parameters['classes'], title = 'Confusion matrix')
        plt.show()
    
    def predict(self, data_path, task):
        
        dataset = pd.read_table(data_path, sep='\t')
        dataset['raw_tweet'] = dataset['tweet']
        dataset['tweet'] = hp.process_tweets(dataset['tweet'])
        test_sequences = self.hparams['tokenizer'].texts_to_sequences(dataset['tweet'])
        test_data_sequence = pad_sequences(test_sequences, maxlen = 50)
        predictions = self.model.predict(test_data_sequence)
        predictions_round = [np.round(x) for x in predictions]
        out = np.concatenate(predictions_round).ravel()
        if task == 'subtask_a':
            dataset['label'] = ["NOT" if x == 0 else "OFF" for x in out]
        if task == 'subtask_b':
            dataset['label'] = ["TIN" if x == 0 else "UNT" for x in out]
        if task == 'subtask_c':
            class_labels = np.argmax(np.array(predictions), axis=1)
            out = np.concatenate(predictions).ravel()
            dataset['label'] = ["IND" if x == 1 else "GRP" if x == 0 else "OTH" for x in class_labels]
            
        return dataset

if __name__ == "__main__":
    
    ###########################################################################
    # Environment settings
    ###########################################################################
    
    environment = {'SMOTE_flag':                            False,
                   'class_weights_flag':                    False,
                   'cross_validation_flag':                 False,
                   'build_final_model_flag':                True,
                   'predict_test_set_flag':                 False,
                   'task':                                  'subtask_a',
                   }
    
    model_parameters = {'epochs':                           3,
                        'classes':                          None,
                        'num_classes':                      None,
                        'tokenizer':                        None,
                        'vocab_size':                       None,
                        'optimizer':                       'adam',
                        'rnn_layer_after_cnn':              False,
                        'rnn_layer':                       'Bidirectional GRU',
                        'use_pretrained_embedding':         True,
                        'convolutional_layer':              True,
                        'task':                             environment['task'],
                        'custom_metrics_f1':                False,
                       }
    
    
    ###########################################################################
    # Data loading and concatenation
    ###########################################################################
    
    train_data = pd.read_table("./start-kit/training-v1/offenseval-training-v1.tsv", sep='\t')
    train_data.drop(['id'], axis = 1, inplace = True)
    
    trial_data = pd.read_table("./start-kit/trial-data/offenseval-trial.txt", sep='\t', header=None, names=['tweet', 'subtask_a', 'subtask_b', 'subtask_c'])
    trial_data['subtask_b'][trial_data['subtask_b'] == 'TTH'] = "TIN"
    trial_data['subtask_c'][trial_data['subtask_c'] == 'ORG'] = "OTH"
    
    overall_dataset = pd.concat([train_data, trial_data])
    overall_dataset.reset_index(drop=True, inplace=True)
    overall_dataset['subtask_a'].value_counts()
    overall_dataset['subtask_b'].value_counts()
    overall_dataset['subtask_c'].value_counts()
    
    ###########################################################################
    # task A + B + C
    ###########################################################################
    
    # Data processing
    overall_dataset.dropna(subset = [environment['task']], inplace = True, axis = 0)
    overall_dataset['raw_tweet'] = overall_dataset['tweet']
    overall_dataset['tweet'] = hp.process_tweets(overall_dataset['tweet'])
    train_labels = overall_dataset[environment['task']]
    model_parameters['classes'] = np.unique(train_labels)
    model_parameters['num_classes'] = len(np.unique(train_labels))
    
    train_data, encoded_train_labels, tokenizer, size_of_vocab = hp.data_preparation(overall_dataset['tweet'], train_labels, model_parameters['num_classes'])
    model_parameters['tokenizer'] = tokenizer
    model_parameters['vocab_size'] = size_of_vocab
        
    # Train test split (80/20)
    print('\n')
    print("Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(train_data, encoded_train_labels, test_size = 0.2, random_state = 2)
    
    # K-fold
    kfold = KFold(n_splits = 5, shuffle = True, random_state = 2)
    skf = StratifiedKFold(n_splits = 5, random_state = 2, shuffle = True)
    #skf.get_n_splits(X, y)
    
    if environment['SMOTE_flag'] == True:
        print("To counter imbalanced dataset: SMOTE Oversampling...")
        sm = SMOTE(random_state=12)
        X_train, y_train = sm.fit_sample(X_train, y_train)
    
    # Class weights
    if environment['class_weights_flag'] == True:
        print("To counter imbalanced dataset: Balancing class weights...")
        if model_parameters['num_classes'] == 2:
            cw = compute_class_weight("balanced", np.unique(y_train), y_train)
        elif model_parameters['num_classes'] > 2:
            y_integers = np.argmax(y_train, axis=1)
            cw = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        weights = dict(enumerate(cw))
    
    # Instantiate classifier
    print('\n')
    print("Initialising classifier...")
    OFF_Detector = Offensive_Detector(model_parameters)
    
    # Wrapping our classifier with Keras wrapper + Cross validation
    if environment['cross_validation_flag'] == True:
        print('\n')
        print("Performing cross validation...")
        estimator = KerasClassifier(build_fn = OFF_Detector.build, epochs = model_parameters['epochs'], verbose = 1)
        scoring = {'acc': 'accuracy',
                   'f1_macro': 'f1_macro'}
        if environment['class_weights_flag'] == True:
            results = cross_validate(estimator, X_train, y_train, cv = skf, scoring = scoring, fit_params={'class_weight': weights}, return_estimator = True)
        else:
            results = cross_validate(estimator, X_train, y_train, cv = skf, scoring = scoring, return_estimator = True)
        print("Average accuracy: %.2f" % results['test_acc'].mean())
        print("Average F1_macro: %.2f" % results['test_f1_macro'].mean())
#    else:
#        OFF_Detector.build()
#        es = [EarlyStopping(monitor='val_f1_macro', patience = 3, mode='max', verbose = 1)]
#        if environment['class_weights_flag'] == True:
#            OFF_Detector.train(X_train, y_train, weights, es)
#        else:
#            OFF_Detector.train(X_train, y_train, es)
#        OFF_Detector.evaluate(X_test, y_test)
    
    #Finalise model and train and evaluate
    if environment['build_final_model_flag'] == True:
        print('\n')
        print("Building final model...")
        OFF_Detector.build()
        if environment['class_weights_flag'] == True:
            OFF_Detector.train(X_train, y_train, weights)
        else:
            OFF_Detector.train(X_train, y_train)
        OFF_Detector.evaluate(X_test, y_test)
    
    if environment['predict_test_set_flag'] == True:
        if environment['task'] == 'subtask_a':
            print('\n')
            print("Predicting test set A for submission...")
            data = OFF_Detector.predict("./Test A Release/testset-taska.tsv", environment['task'])
            data[['id', 'label']].to_csv('task_A_submission4.csv', index=False)
        if environment['task'] == 'subtask_b':
            print('\n')
            print("Predicting test set B for submission...")
            data = OFF_Detector.predict("./Test B Release/testset-taskb.tsv", environment['task'])
            data[['id', 'label']].to_csv('task_B_submission3.csv', index=False)
        if environment['task'] == 'subtask_c':
            print('\n')
            print("Predicting test set C for submission...")
            data = OFF_Detector.predict("./Test C Release/test_set_taskc.tsv", environment['task'])
            data[['id', 'label']].to_csv('task_C_submission3.csv', index=False)
    
#    Grid search
#    optimizers = ['rmsprop', 'adam']
#    epochs = [3, 5, 10]
#    param_grid = dict(optimizer = optimizers, epochs = epochs)
#    grid = GridSearchCV(estimator = estimator, param_grid = param_grid)
#    grid_result = grid.fit(X_train, y_train)
#    
#    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#    means = grid_result.cv_results_['mean_test_score']
#    stds = grid_result.cv_results_['std_test_score']
#    params = grid_result.cv_results_['params']
#    for mean, stdev, param in zip(means, stds, params):
#        print("%f (%f) with: %r" % (mean, stdev, param))


# character level
# bert
# ELMO embedding - contextualised word embeddings