import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import cv2 # OpenCV per il ridimensionamento, se necessario

# Versione di TensorFlow
print(f"TensorFlow Version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Attualmente, imposta la crescita della memoria come necessaria per evitare errori di memoria Out-of-Memory (OOM)
        # su alcune GPU limitando l'allocazione di memoria all'inizio.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Sono state trovate {len(gpus)} GPU:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    except RuntimeError as e:
        # La crescita della memoria deve essere impostata prima che le GPU siano state inizializzate
        print(e)
else:
    print("Nessuna GPU trovata. TensorFlow utilizzerà la CPU.")
    print("Verifica l'installazione dei driver NVIDIA, CUDA Toolkit e cuDNN.")

## 1. Configurazione e Variabili Globali
# --------------------------------------------------
# Definisci i percorsi principali per i tuoi dati
BASE_PATH = "./" # Modifica se i tuoi dati sono altrove
TRAIN_DIR = os.path.join(BASE_PATH, "train/")
TEST_DIR = os.path.join(BASE_PATH, "test/")
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")

# Parametri per le immagini e il modello
IMG_HEIGHT = 64  # Altezza desiderata per le immagini
IMG_WIDTH = 64   # Larghezza desiderata per le immagini
IMG_CHANNELS = 3 # M, N, T
BATCH_SIZE = 16     #->32 ben
EPOCHS = 50      # Numero di epoche (da ottimizzare) ->200 ben
RANDOM_STATE = 17 # Per la riproducibilità ->42ben

# Imposta il seed per la riproducibilità
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

## 2. Caricamento Metadati
# --------------------------------------------------
try:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    print("Train CSV caricato con successo:")
    print(train_df.head())
    print(f"\nNumero totale di campioni di addestramento: {len(train_df)}")
    print(f"Distribuzione delle classi nel set di addestramento:\n{train_df['class'].value_counts(normalize=True)}")
except FileNotFoundError:
    print(f"ERRORE: File {TRAIN_CSV_PATH} non trovato. Assicurati che il percorso sia corretto.")
    # Potresti voler arrestare lo script qui o gestire l'errore diversamente
    # Per questo template, continuiamo assumendo che il file esista.
    # train_df = pd.DataFrame({'folder': [], 'class': []}) # Placeholder

## 3. Preparazione dei Dati e Data Generator
# --------------------------------------------------
# Funzione per caricare e pre-processare un'immagine (una cellula)
def load_cell_images(cell_id, base_dir):
    """
    Carica i tre canali per una data cellula e li combina.
    Normalizza e ridimensiona le immagini.
    """
    channels = []
    for channel_suffix in ['_M.npy', '_N.npy', '_T.npy']:
        file_path = os.path.join(base_dir, cell_id, cell_id + channel_suffix)
        try:
            img_array = np.load(file_path)
            # Ridimensionamento (se necessario, altrimenti assicurati che siano tutti della stessa dimensione)
            # Qui usiamo cv2.resize. Potresti usare tf.image.resize con tf.Tensor.
            if img_array.shape[0] != IMG_HEIGHT or img_array.shape[1] != IMG_WIDTH:
                img_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            
            # Normalizzazione dei pixel a [0, 1]
            img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-6) # Aggiungi epsilon per evitare divisione per zero
            
            channels.append(img_array)
        except FileNotFoundError:
            print(f"Attenzione: File non trovato {file_path}")
            # Restituisci un array vuoto o gestisci l'errore come preferisci
            return None
        except Exception as e:
            print(f"Errore durante il caricamento o pre-processamento di {file_path}: {e}")
            return None

    if len(channels) == 3:
        # Impila i canali per formare un'immagine HxWxC (es. 64x64x3)
        return np.stack(channels, axis=-1)
    else:
        return None

# Classe Keras Data Generator
class CellDataGenerator(keras.utils.Sequence):
    def __init__(self, cell_ids, labels, batch_size, base_dir, dim=(IMG_HEIGHT, IMG_WIDTH), n_channels=IMG_CHANNELS, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.cell_ids = cell_ids
        self.n_channels = n_channels
        self.base_dir = base_dir
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.cell_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Genera indici del batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Trova la lista degli ID delle cellule
        batch_cell_ids = [self.cell_ids[k] for k in indexes]
        
        # Genera dati
        X, y = self.__data_generation(batch_cell_ids, indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.cell_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_cell_ids, batch_indexes):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generazione dei dati
        for i, cell_id in enumerate(batch_cell_ids):
            img_data = load_cell_images(cell_id, self.base_dir)
            if img_data is not None:
                X[i,] = img_data
                if self.labels is not None: # Per il set di test, labels sarà None
                    y[i] = self.labels[batch_indexes[i]]
            else:
                # Gestisci il caso in cui l'immagine non può essere caricata
                # Potresti usare un'immagine placeholder o saltare questo campione
                # Qui, per semplicità, usiamo zeri, ma dovresti considerare una strategia migliore
                print(f"Attenzione: Impossibile caricare i dati per {cell_id}. Verrà usato un array di zeri.")
                X[i,] = np.zeros((*self.dim, self.n_channels)) 
                if self.labels is not None:
                    y[i] = self.labels[batch_indexes[i]] # Mantieni l'etichetta corretta se possibile

        return X, y

# Divisione dei dati di addestramento in set di addestramento e validazione
if 'folder' in train_df.columns and 'class' in train_df.columns:
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        train_df['folder'].values,
        train_df['class'].values,
        test_size=0.2,       # 20% per la validazione
        random_state=RANDOM_STATE,
        stratify=train_df['class'].values # Stratifica per mantenere la proporzione delle classi
    )

    print(f"Cellule per addestramento: {len(train_ids)}")
    print(f"Cellule per validazione: {len(val_ids)}")

    # Creazione dei generatori
    train_generator = CellDataGenerator(train_ids, train_labels, BATCH_SIZE, TRAIN_DIR)
    val_generator = CellDataGenerator(val_ids, val_labels, BATCH_SIZE, TRAIN_DIR, shuffle=False) # Non mischiare il validation set
else:
    print("ERRORE: Le colonne 'folder' o 'class' non sono presenti in train_df. Impossibile creare i generatori.")
    train_generator = None
    val_generator = None


## 4. Visualizzazione di Esempio (Opzionale)
# --------------------------------------------------
if train_generator:
    # Visualizza un batch di immagini dal generatore di addestramento
    sample_batch_images, sample_batch_labels = train_generator[0] # Prendi il primo batch

    plt.figure(figsize=(12, 12))
    for i in range(min(9, len(sample_batch_images))): # Mostra fino a 9 immagini
        plt.subplot(3, 3, i + 1)
        # Mostriamo solo il primo canale (Membrana) per semplicità o una combinazione
        # Per mostrare un'immagine RGB, assicurati che i valori siano in [0,1] o [0,255]
        # Qui assumiamo che l'immagine sia HxWxC e i valori siano già normalizzati a [0,1]
        img_display = sample_batch_images[i]
        if img_display.shape[-1] == 1: # Se è in scala di grigi
             plt.imshow(img_display[:,:,0], cmap='gray')
        else: # Se ha più canali, puoi mostrarne uno o combinarli
             plt.imshow(img_display) # Matplotlib gestisce bene HxWx3 con valori in [0,1]
        plt.title(f"Classe: {sample_batch_labels[i]}")
        plt.axis("off")
    plt.suptitle("Esempi di Immagini dal Training Set (Canali Combinati)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Verifica le dimensioni di un'immagine di output dal generatore
    print(f"Shape di un batch di immagini: {sample_batch_images.shape}")
    print(f"Shape di un batch di etichette: {sample_batch_labels.shape}")
    print(f"Shape di una singola immagine pre-processata: {sample_batch_images[0].shape}")
else:
    print("Skipping visualizzazione: train_generator non inizializzato.")

## 5. Definizione del Modello CNN
# --------------------------------------------------
def create_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25), # Aggiunto Dropout

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25), # Aggiunto Dropout

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25), # Aggiunto Dropout
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Dropout più aggressivo prima del layer finale
        layers.Dense(1, activation='sigmoid') # Output binario (0 o 1) con sigmoide
    ])
    return model

model = create_cnn_model()
model.summary()

## 6. Compilazione e Addestramento del Modello
# --------------------------------------------------
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), # Adam è una buona scelta di default
              loss='binary_crossentropy',      # Per classificazione binaria
              metrics=['accuracy'])            # Metrica da massimizzare

# Callbacks (Opzionale ma raccomandato)
# Salva il modello migliore e interrompi l'addestramento se non ci sono miglioramenti
model_checkpoint = keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
#early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, mode='max') # Aumenta la pazienza se necessario

if train_generator and val_generator:
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[model_checkpoint] #, early_stopping]
    )

    # Visualizzazione delle curve di apprendimento
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy Addestramento')
    plt.plot(history.history['val_accuracy'], label='Accuracy Validazione')
    plt.title('Accuracy del Modello')
    plt.xlabel('Epoca')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss Addestramento')
    plt.plot(history.history['val_loss'], label='Loss Validazione')
    plt.title('Loss del Modello')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Carica il modello migliore salvato da ModelCheckpoint
    print("Caricamento del modello migliore da 'best_model.keras'...")
    model = keras.models.load_model('best_model.keras')

    # Valutazione finale sul set di validazione (con il modello migliore)
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"\nPerformance finale sul set di validazione:")
    print(f"Loss: {val_loss:.4f}")
    print(f"Accuratezza: {val_accuracy:.4f}")

else:
    print("ERRORE: Impossibile avviare l'addestramento, i generatori di dati non sono stati creati.")


## 7. Preparazione Dati di Test e Predizione
# --------------------------------------------------
# Elenca tutti gli ID delle cellule nel set di test (nomi delle cartelle)
try:
    test_cell_ids = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    test_cell_ids.sort() # Assicurati un ordine consistente se necessario
    print(f"Trovate {len(test_cell_ids)} cellule nel set di test.")
    print(f"Primi 5 ID del test set: {test_cell_ids[:5]}")
except FileNotFoundError:
    print(f"ERRORE: Cartella di test {TEST_DIR} non trovata.")
    test_cell_ids = []

if test_cell_ids:
    # Crea un generatore per il test set (senza etichette)
    # Nota: la classe CellDataGenerator può gestire labels=None
    # Per l'ultimo batch che potrebbe essere più piccolo di batch_size, si può gestire o usare batch_size=1 per la predizione
    # Oppure, si può creare un array di immagini e predire su quello.
    # Per semplicità, useremo un approccio che predice cellula per cellula se la gestione dei batch parziali diventa complessa.
    
    predictions = []
    
    # Metodo 1: Predire cellula per cellula (più semplice da implementare, ma meno efficiente)
    # for cell_id in test_cell_ids:
    #     img = load_cell_images(cell_id, TEST_DIR)
    #     if img is not None:
    #         img_batch = np.expand_dims(img, axis=0) # Crea un batch di 1 immagine
    #         pred_prob = model.predict(img_batch, verbose=0)[0]
    #         pred_class = 1 if pred_prob > 0.5 else 0
    #         predictions.append({'ID': cell_id, 'prediction': pred_class})
    #     else:
    #         print(f"Attenzione: Impossibile caricare {cell_id} dal test set. Predizione impostata a 0 (o gestisci diversamente).")
    #         predictions.append({'ID': cell_id, 'prediction': 0}) # Fallback

    # Metodo 2: Utilizzare un generatore per il test set (più efficiente)
    # È necessario modificare leggermente CellDataGenerator per gestire il caso in cui len(cell_ids) non è divisibile per batch_size
    # o creare un ciclo di predizione più attento.
    # Qui creiamo un semplice array di test. Se la memoria è un problema, usa un generatore.
    
    X_test = []
    valid_test_ids = [] # Per tenere traccia degli ID per cui abbiamo effettivamente caricato i dati
    for cell_id in test_cell_ids:
        img = load_cell_images(cell_id, TEST_DIR)
        if img is not None:
            X_test.append(img)
            valid_test_ids.append(cell_id)
        else:
            print(f"Attenzione: Impossibile caricare {cell_id} dal test set. Verrà saltato.")
            # Potresti voler aggiungere una predizione di default per questi ID mancanti nel file di submission
            # Questo template assume che tutti gli ID nel CSV di submission devono avere una predizione.
            # Per ora, lo saltiamo, ma in una competizione dovresti gestirlo.

    if X_test:
        X_test = np.array(X_test)
        print(f"Shape dell'array di test X_test: {X_test.shape}")
        
        test_predictions_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
        test_predictions_classes = (test_predictions_probs > 0.5).astype(int).flatten()

        # Assicurati che il numero di predizioni corrisponda agli ID validi
        if len(test_predictions_classes) == len(valid_test_ids):
            for i in range(len(valid_test_ids)):
                predictions.append({'ID': valid_test_ids[i], 'prediction': test_predictions_classes[i]})
        else:
            print("ERRORE: Discrepanza tra numero di predizioni e ID validi.")
            # Fallback: riempi con predizioni di default se necessario per tutti i test_cell_ids
            # Questo scenario non dovrebbe accadere con il codice sopra se tutti gli X_test sono validi.
            # Se alcuni ID sono stati saltati, devi decidere come gestirli nel file di submission.
            # Per ora, il file di submission conterrà solo gli ID validi.
            # Se TUTTI gli ID originali devono essere presenti, dovrai fare un merge.

    else:
        print("Nessun dato di test valido caricato. Impossibile fare predizioni.")


## 8. Generazione del File di Submission
# --------------------------------------------------
if predictions:
    submission_df = pd.DataFrame(predictions)
    # Assicurati che le colonne siano nell'ordine corretto ('ID', 'prediction')
    submission_df = submission_df[['ID', 'prediction']]
    
    # Ordina per ID se necessario (dipende dai requisiti della submission)
    # submission_df = submission_df.sort_values(by='ID') 
    
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nFile di submission generato: {submission_path}")
    print(submission_df.head())
else:
    print("\nNessuna predizione generata. Il file di submission non sarà creato.")
    # Se devi comunque creare un file con tutti gli ID di test e predizioni di default:
    # submission_df = pd.DataFrame({'ID': test_cell_ids, 'prediction': 0}) # Esempio: predici sempre 0
    # submission_df.to_csv("submission_default.csv", index=False)
    # print("File di submission di default creato con predizioni a 0.")

print("\nPipeline completata.")

## 9. Possibili Miglioramenti e Passi Successivi
# --------------------------------------------------
# - **Ottimizzazione degli Iperparametri**: Prova diverse architetture di CNN, learning rate, batch size, numero di epoche.
# - **Data Augmentation**: Aggiungi data augmentation al `CellDataGenerator` (rotazioni, flip, zoom, etc.) per migliorare la generalizzazione del modello.
#   Puoi usare `tf.keras.preprocessing.image.ImageDataGenerator` o `tf.image` per questo.
# - **Transfer Learning**: Sperimenta con modelli pre-addestrati (es. MobileNetV2, ResNet50) se le dimensioni delle immagini e il dataset lo consentono.
#   Richiederebbe di adattare l'input del modello pre-addestrato.
# - **Analisi degli Errori**: Esamina i casi in cui il modello sbaglia per capire se ci sono pattern.
# - **Gestione Avanzata dei Canali**: Invece di impilarli semplicemente, potresti provare architetture che processano i canali separatamente all'inizio e poi fondono le feature.
# - **Controllo Dimensioni Immagini**: Verifica le dimensioni originali delle immagini .npy. Se sono molto variabili, il ridimensionamento potrebbe perdere informazioni.
#   Se sono molto grandi, potresti aver bisogno di più memoria o di usare patch.
# - **Normalizzazione**: Sperimenta con diverse tecniche di normalizzazione. Quella usata (min-max per immagine) è una scelta comune.
# - **Cross-Validation**: Per una stima più robusta delle performance, usa la cross-validation.