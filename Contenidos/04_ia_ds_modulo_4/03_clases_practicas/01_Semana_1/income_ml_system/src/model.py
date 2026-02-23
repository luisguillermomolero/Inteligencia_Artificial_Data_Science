import tensorflow as tf

def build_model(input_dim: int) -> tf.keras.Model:
    
    model = tf.keras.Sequential([
        
        tf.keras.layers.Input(shape=(input_dim,)),
        
        tf.keras.layers.Dense(64, activation="relu"),
        
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(32, activation="relu"),
        
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        
        loss="binary_crossentropy",
        
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    
    return model