import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time

import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
from tensorflow.keras.optimizers import AdamW


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model: tf.keras.Model | None = None
        self.train_generator: tf.keras.preprocessing.image.DirectoryIterator | None = None
        self.valid_generator: tf.keras.preprocessing.image.DirectoryIterator | None = None

    # -----------------------
    # Load base model
    # -----------------------
    def get_base_model(self) -> tf.keras.Model:
        path = str(self.config.updated_base_model_path)
        if not path.endswith(".keras") and not path.endswith(".h5"):
            path += ".keras"
        self.model = tf.keras.models.load_model(path)
        print(f"✅ Model loaded from: {path}")
        return self.model

    # -----------------------
    # Unfreeze last N layers
    # -----------------------
    def unfreeze_top_layers(self, num_layers: int = None, learning_rate: float = None):
        if self.model is None:
            raise ValueError("Load the model first using get_base_model()")

        if num_layers is None:
            num_layers = self.config.params_fine_tune_last_n
        if learning_rate is None:
            learning_rate = self.config.params_learning_rate

        # Unfreeze selected layers
        for layer in self.model.layers[-num_layers:]:
            layer.trainable = True

        print(f"✅ Top {num_layers} layers set to trainable for fine-tuning.")

        # Recompile model
        self.model.compile(
            optimizer=AdamW(
                learning_rate=learning_rate,
                weight_decay=self.config.params_weight_decay
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=self.config.params_label_smoothing
            ),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall")
            ]
        )
        print(f"✅ Model recompiled after unfreezing layers (lr={learning_rate}).")

    # -----------------------
    # Generators
    # -----------------------
    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.2,
        )
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:2],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical"
        )

        # Validation generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            str(self.config.training_data),
            subset="validation",
            shuffle=False,
            seed=42,
            **dataflow_kwargs
        )

        # Training generator with augmentation
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=20,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=False,
                brightness_range=[0.8, 1.2],
                fill_mode="nearest",
                validation_split=0.2
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            str(self.config.training_data),
            subset="training",
            shuffle=True,
            seed=42,
            **dataflow_kwargs
        )

    # -----------------------
    # Callbacks
    # -----------------------
    def get_callbacks(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            mode="max",
            verbose=1
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.2,
            patience=3,
            verbose=1,
            min_lr=1e-7
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.trained_model_path),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        )
        return [early_stopping, reduce_lr, checkpoint]

    # -----------------------
    # Training with progressive unfreezing
    # -----------------------
    def train(self):
        if self.model is None:
            raise ValueError("Model not loaded. Call get_base_model() first.")
        if self.train_generator is None or self.valid_generator is None:
            raise ValueError("Generators not prepared. Call train_valid_generator() first.")

        callbacks = self.get_callbacks()

        # Progressive unfreezing schedule (layers, learning_rate)
        unfreeze_schedule = [
            (10, 1e-4),   # Phase 1
            (20, 5e-5),   # Phase 2
            (40, 1e-5),   # Phase 3
        ]

        history = None
        for i, (layers, lr) in enumerate(unfreeze_schedule, 1):
            print(f"\n🚀 Phase {i}: Unfreezing last {layers} layers with lr={lr}...")
            self.unfreeze_top_layers(num_layers=layers, learning_rate=lr)
            history = self.model.fit(
                self.train_generator,
                validation_data=self.valid_generator,
                epochs=5,  # per phase
                callbacks=callbacks
            )

        # Save final model (best checkpoint already saved)
        self.save_model(self.config.trained_model_path, self.model)
        return history

    # -----------------------
    # Save model
    # -----------------------
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path_str = str(path)
        if not path_str.endswith(".keras") and not path_str.endswith(".h5"):
            path_str += ".keras"
        model.save(path_str, include_optimizer=False)
        print(f"✅ Model saved at: {path_str}")
