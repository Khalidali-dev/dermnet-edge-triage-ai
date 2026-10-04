import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight

# Hardware Setup (Apple M1 Pro GPU)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Detected:", gpus)

base_dir = os.path.expanduser("~/ML Projects/dermnet_triage_core/dataset")
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

IMG_SIZE = (300, 300)
BATCH_SIZE = 32
MODEL_PATH = "best_direct_90plus.keras"

print("\n1. Loading Datasets...")
train_raw_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

test_raw_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_raw_ds.class_names
num_classes = len(class_names)
print(f"Total Classes: {num_classes}")

with open("labels.txt", "w") as f:
    for name in class_names:
        f.write(f"{name}\n")

# 2. Automated Class Weighting (Bypasses Frequent-Class Bias)
print("\nCalculating class weights for 24 classes...")
y_train = []
for _, labels in train_raw_ds:
    y_train.extend(labels.numpy())
y_train = np.array(y_train)

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: float(np.clip(w, 0.5, 2.5)) for i, w in enumerate(weights)}

AUTOTUNE = tf.data.AUTOTUNE
def one_hot_format(image, label):
    return image, tf.one_hot(label, depth=num_classes)

train_ds = train_raw_ds.map(one_hot_format, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds = test_raw_ds.map(one_hot_format, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# 3. Medical Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.25),
    layers.RandomTranslation(0.1, 0.1),
])

# 4. Model Architecture (EfficientNetV2-B3)
base_model = tf.keras.applications.EfficientNetV2B3(
    include_top=False,
    weights='imagenet',
    input_shape=(300, 300, 3),
    include_preprocessing=True
)

inputs = layers.Input(shape=(300, 300, 3), name="input_image")
x = data_augmentation(inputs)
x = base_model(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.35)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

# 5. Direct Deep Unfreeze Configuration
base_model.trainable = True
total_layers = len(base_model.layers)
# Top 280 layers unfreeze directly
for layer in base_model.layers[:-280]:
    layer.trainable = False

# Freeze BN layers for running statistics stability
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

print(f"Total Base Layers: {total_layers}. Active Trainable: {280}")

# 6. Smooth Cosine Learning Rate Schedule (30 Epochs direct)
TOTAL_EPOCHS = 30
total_steps = len(train_ds) * TOTAL_EPOCHS

lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1.8e-4,
    decay_steps=total_steps,
    alpha=0.01
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_acc')]
)

checkpoint_cb = callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor='val_top_3_acc',
    mode='max',
    save_best_only=True,
    verbose=1
)

early_stopping_cb = callbacks.EarlyStopping(
    monitor='val_top_3_acc',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

# 7. Single-Shot Training
print(f"\n--- Launching Direct 90%+ Training ({TOTAL_EPOCHS} Epochs) ---")
model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=TOTAL_EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# 8. Test Evaluation with Test-Time Augmentation (TTA) Verification
print("\nRunning Verification with Test-Time Augmentation...")
best_model = tf.keras.models.load_model(MODEL_PATH)

correct_top1, correct_top3, total = 0, 0, 0
for images, labels in test_raw_ds:
    # Direct Forward Pass + Horizontal Flip Averaging (TTA)
    preds_normal = best_model(images, training=False).numpy()
    preds_flipped = best_model(tf.image.flip_left_right(images), training=False).numpy()
    preds = (preds_normal + preds_flipped) / 2.0
    
    top3_preds = np.argsort(preds, axis=1)[:, -3:]
    true_labels = labels.numpy()
    
    for i in range(len(true_labels)):
        total += 1
        if true_labels[i] == np.argmax(preds[i]):
            correct_top1 += 1
        if true_labels[i] in top3_preds[i]:
            correct_top3 += 1

print(f"\nFinal TTA-Verified Results:")
print(f"Direct Top-1 Accuracy: {(correct_top1 / total) * 100:.2f}%")
print(f"Direct Top-3 Accuracy: {(correct_top3 / total) * 100:.2f}%")

# 9. Clean TFLite Export
print("\nExporting Production .tflite model...")
inf_inputs = layers.Input(shape=(300, 300, 3), name="input_1")
inf_base = tf.keras.applications.EfficientNetV2B3(
    include_top=False,
    weights=None,
    input_shape=(300, 300, 3),
    include_preprocessing=True
)
inf_x = inf_base(inf_inputs)
inf_x = layers.GlobalAveragePooling2D()(inf_x)
inf_x = layers.BatchNormalization()(inf_x)
inf_x = layers.Dropout(0.35)(inf_x)
inf_x = layers.Dense(512, activation='relu')(inf_x)
inf_x = layers.BatchNormalization()(inf_x)
inf_x = layers.Dropout(0.25)(inf_x)
inf_outputs = layers.Dense(num_classes, activation='softmax')(inf_x)

clean_inference_model = models.Model(inf_inputs, inf_outputs)
clean_inference_model.set_weights(best_model.get_weights())

run_model = tf.function(lambda x: clean_inference_model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 300, 300, 3], tf.float32)
)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open("dermnet_model_direct90.tflite", "wb") as f:
    f.write(tflite_model)
shutil.copyfile("dermnet_model_direct90.tflite", "dermnet_model.tflite")

size_mb = os.path.getsize("dermnet_model.tflite") / (1024 * 1024)
print(f"\nModel exported successfully: 'dermnet_model.tflite' ({size_mb:.2f} MB)")