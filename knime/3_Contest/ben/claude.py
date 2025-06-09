import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# Configurazione GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU trovate: {len(gpus)}")
    except RuntimeError as e:
        print(e)

## CONFIGURAZIONE
BASE_PATH = "./"
TRAIN_DIR = os.path.join(BASE_PATH, "train/")
TEST_DIR = os.path.join(BASE_PATH, "test/")
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")

IMG_HEIGHT = 96  # Compromesso tra qualità e velocità
IMG_WIDTH = 96   
IMG_CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 80
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

## CARICAMENTO DATI
train_df = pd.read_csv(TRAIN_CSV_PATH)
print(f"Dataset: {len(train_df)} campioni")
print(f"Distribuzione classi:\n{train_df['class'].value_counts(normalize=True)}")

## PREPROCESSING SEMPLIFICATO
def load_and_preprocess_cell(cell_id, base_dir):
    """Caricamento semplificato senza OpenCV"""
    channels = []
    channel_names = ['_M.npy', '_N.npy', '_T.npy']
    
    for channel_suffix in channel_names:
        file_path = os.path.join(base_dir, cell_id, cell_id + channel_suffix)
        try:
            img_array = np.load(file_path).astype(np.float32)
            
            # Ridimensionamento con TensorFlow
            img_tensor = tf.constant(img_array)
            if len(img_tensor.shape) == 2:
                img_tensor = tf.expand_dims(img_tensor, -1)
            img_resized = tf.image.resize(img_tensor, [IMG_HEIGHT, IMG_WIDTH])
            img_array = img_resized.numpy().squeeze()
            
            # Normalizzazione robusta
            p1, p99 = np.percentile(img_array, [1, 99])
            img_array = np.clip(img_array, p1, p99)
            img_array = (img_array - p1) / (p99 - p1 + 1e-8)
            
            channels.append(img_array)
            
        except Exception as e:
            print(f"Errore caricamento {file_path}: {e}")
            return None
    
    if len(channels) == 3:
        return np.stack(channels, axis=-1)
    return None

## DATA GENERATOR SEMPLIFICATO
class SimpleCellDataGenerator(keras.utils.Sequence):
    def __init__(self, cell_ids, labels, batch_size, base_dir, 
                 dim=(IMG_HEIGHT, IMG_WIDTH), n_channels=IMG_CHANNELS, 
                 shuffle=True, augment=False):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.cell_ids = cell_ids
        self.n_channels = n_channels
        self.base_dir = base_dir
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.cell_ids) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.cell_ids))
        batch_size_actual = end_idx - start_idx
        
        indexes = self.indexes[start_idx:end_idx]
        batch_cell_ids = [self.cell_ids[k] for k in indexes]
        
        X = np.empty((batch_size_actual, *self.dim, self.n_channels))
        y = np.empty((batch_size_actual), dtype=int)
        
        for i, cell_id in enumerate(batch_cell_ids):
            img_data = load_and_preprocess_cell(cell_id, self.base_dir)
            if img_data is not None:
                if self.augment:
                    img_data = self.simple_augmentation(img_data)
                X[i,] = img_data
                if self.labels is not None:
                    y[i] = self.labels[indexes[i]]
            else:
                X[i,] = np.zeros((*self.dim, self.n_channels))
                if self.labels is not None:
                    y[i] = self.labels[indexes[i]]
        
        return X, y if self.labels is not None else X
    
    def simple_augmentation(self, img):
        """Augmentation semplice con TensorFlow"""
        img_tensor = tf.constant(img)
        
        # Flip random
        if np.random.random() > 0.5:
            img_tensor = tf.image.flip_left_right(img_tensor)
        if np.random.random() > 0.5:
            img_tensor = tf.image.flip_up_down(img_tensor)
        
        # Rotazione leggera (solo 90°, 180°, 270°)
        if np.random.random() > 0.7:
            k = np.random.choice([1, 2, 3])
            img_tensor = tf.image.rot90(img_tensor, k=k)
        
        # Brightness/contrast
        if np.random.random() > 0.8:
            img_tensor = tf.image.random_brightness(img_tensor, 0.1)
            img_tensor = tf.image.random_contrast(img_tensor, 0.9, 1.1)
        
        return img_tensor.numpy()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.cell_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

## MODELLO OTTIMIZZATO
def create_efficient_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    """Modello efficiente ma potente"""
    inputs = keras.Input(shape=input_shape)
    
    # Initial conv
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Block 1
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2  
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classifier
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    return model

## TRAINING
# Calcolo manuale class weights
class_counts = train_df['class'].value_counts().sort_index()
total_samples = len(train_df)
class_weight_dict = {
    0: total_samples / (2 * class_counts[0]),
    1: total_samples / (2 * class_counts[1])
}
print(f"Class weights: {class_weight_dict}")

# Split manuale stratificato
def stratified_split(df, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    
    # Separa per classe
    class_0 = df[df['class'] == 0]
    class_1 = df[df['class'] == 1]
    
    # Calcola dimensioni
    n_val_0 = int(len(class_0) * test_size)
    n_val_1 = int(len(class_1) * test_size)
    
    # Campiona randomly
    val_idx_0 = np.random.choice(class_0.index, n_val_0, replace=False)
    val_idx_1 = np.random.choice(class_1.index, n_val_1, replace=False)
    
    val_indices = np.concatenate([val_idx_0, val_idx_1])
    train_indices = df.index[~df.index.isin(val_indices)]
    
    return train_indices, val_indices

train_indices, val_indices = stratified_split(train_df)

train_ids = train_df.loc[train_indices, 'folder'].values
train_labels = train_df.loc[train_indices, 'class'].values
val_ids = train_df.loc[val_indices, 'folder'].values  
val_labels = train_df.loc[val_indices, 'class'].values

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

# Generatori
train_generator = SimpleCellDataGenerator(
    train_ids, train_labels, BATCH_SIZE, TRAIN_DIR, 
    shuffle=True, augment=True
)
val_generator = SimpleCellDataGenerator(
    val_ids, val_labels, BATCH_SIZE, TRAIN_DIR, 
    shuffle=False, augment=False
)

# Modello
model = create_efficient_model()
print(f"Parametri del modello: {model.count_params():,}")

# Compilazione
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_model_simple.keras', 
        save_best_only=True, 
        monitor='val_accuracy', 
        mode='max',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=12, 
        restore_best_weights=True, 
        mode='max'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=6,
        min_lr=1e-7,
        verbose=1
    )
]

# Training
print("Avvio training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# Carica miglior modello
model = keras.models.load_model('best_model_simple.keras')

## VALUTAZIONE
val_predictions_probs = model.predict(val_generator)
val_predictions = (val_predictions_probs > 0.5).astype(int).flatten()

# Metriche manuali
def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1, [[tn, fp], [fn, tp]]

acc, prec, rec, f1, cm = calculate_metrics(val_labels, val_predictions)

print("\n" + "="*50)
print("RISULTATI VALIDAZIONE")
print("="*50)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"Confusion Matrix:\n{np.array(cm)}")

## PREDIZIONI TEST
test_cell_ids = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
test_cell_ids.sort()

test_generator = SimpleCellDataGenerator(
    test_cell_ids, None, BATCH_SIZE, TEST_DIR, 
    shuffle=False, augment=False
)

test_predictions_probs = model.predict(test_generator, verbose=1)
test_predictions = (test_predictions_probs > 0.5).astype(int).flatten()

# Correzione lunghezza se necessario
if len(test_predictions) != len(test_cell_ids):
    min_len = min(len(test_predictions), len(test_cell_ids))
    test_predictions = test_predictions[:min_len]
    test_cell_ids = test_cell_ids[:min_len]

# Submission
submission_df = pd.DataFrame({
    'ID': test_cell_ids,
    'prediction': test_predictions
})

submission_df.to_csv('submission_simple.csv', index=False)
print(f"\nSubmission creata: {len(submission_df)} predizioni")
print(f"Distribuzione predizioni: {np.bincount(test_predictions)}")

# Plot risultati
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print(f"\nMiglior validation accuracy: {max(history.history['val_accuracy']):.4f}")
print("Pipeline completata con successo!")