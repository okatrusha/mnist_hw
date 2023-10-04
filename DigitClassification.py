import tensorflow as tf
import keras 
from keras import layers, callbacks
from keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, Flatten, Dropout

class DigitClassification:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        self.model = keras.Sequential([
            Conv2D(64, kernel_size=(3,3), padding='Same', activation='relu', input_shape = (28,28,1)),
            Conv2D(64, kernel_size=(3,3), padding='Same', activation='relu'),
            MaxPooling2D(2,2),
            BatchNormalization(),
            Conv2D(128, kernel_size=(3,3), padding='Same', activation='relu'),
            Conv2D(128, kernel_size=(3,3), padding='Same', activation='relu'),            
            MaxPooling2D(2,2),            
            BatchNormalization(),            

            Conv2D(filters=256, kernel_size=(3,3), activation='relu'),            
            MaxPooling2D(2,2),            
            BatchNormalization(),            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax'),
        ])

        return self.model
    
    def compile_model(self, learning_rate=0.001):
        self.model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
        
    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        '''early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', 
        patience=5,
        verbose=1,
        mode='max',
        restore_best_weights=True)
        '''
        
        history = self.model.fit(x=X_train,
                                    y=y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_data=(X_val, y_val),
                                    #callbacks=[early_stopping]
                                    )
        return history
        
    def evaluate_model(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predict_model(self, X_test):
        return self.model.predict(X_test)
    
    def summary(self):
        return self.model.summary()
        