import os
import time
import numpy as np
import tensorflow as tf
from data_transformation import get_augmentation
from utils import get_project_root, load_config
import mlflow
import mlflow.tensorflow

config = load_config()
CLASSES = config['training']['classes']

def save_feedback_features(feedback_id, label, img_array):
    """
    Applies augmentation to the base preprocessed img_array 
    and saves the original + augmented variants as .npy files.
    """
    features_dir = os.path.join(get_project_root(), "features", label)
    os.makedirs(features_dir, exist_ok=True)
    
    base_filename = os.path.join(features_dir, feedback_id)
    
    # Save original
    np.save(f"{base_filename}_orig.npy", img_array)
    
    # Generate 3 augmented versions
    aug_layer = get_augmentation()
    img_batch = np.expand_dims(img_array, 0)
    
    for i in range(1, 4):
        # training=True ensures Random layers apply transformations
        aug_img = aug_layer(img_batch, training=True)[0].numpy()
        np.save(f"{base_filename}_aug{i}.npy", aug_img)

def load_drift_features():
    """Loads all saved .npy features for finetuning."""
    features_dir = os.path.join(get_project_root(), "features")
    if not os.path.exists(features_dir):
        return None, None
        
    X, y = [], []
    for label in CLASSES:
        class_dir = os.path.join(features_dir, label)
        if not os.path.exists(class_dir):
            continue
            
        class_idx = CLASSES.index(label)
        for file in os.listdir(class_dir):
            if file.endswith(".npy"):
                path = os.path.join(class_dir, file)
                try:
                    arr = np.load(path)
                    X.append(arr)
                    y.append(class_idx)
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                    
    if not X:
        return None, None
        
    return np.array(X), np.array(y)

def run_finetuning(model):
    """
    Finetunes the given model on the newly collected drift features.
    Returns the updated model path.
    """
    X_feat, y_feat = load_drift_features()
    if X_feat is None or len(X_feat) == 0:
        raise ValueError("No drift features found for finetuning.")
        
    print(f"🔧 Finetuning model on {len(X_feat)} new augmented features...")
    
    # Setup MLflow logging for finetuning
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'] + "-Finetuning")
    
    with mlflow.start_run(run_name=f"finetune_{int(time.time())}"):
        mlflow.log_param("num_features", len(X_feat))
        
        # Finetune with a small learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train on features
        history = model.fit(
            X_feat, y_feat,
            epochs=5,
            batch_size=8,
            validation_split=0.2 if len(X_feat) > 10 else 0.0
        )
        
        # Log metrics
        for i in range(len(history.history['accuracy'])):
            mlflow.log_metric("accuracy", history.history['accuracy'][i], step=i)
            mlflow.log_metric("loss", history.history['loss'][i], step=i)
            
        # Overwrite or save new model
        model_dir = os.path.join(get_project_root(), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "fire_model.keras")
        model.save(model_path)
        
        mlflow.tensorflow.log_model(model, artifact_path="model_finetuned")
        
    return model_path
