import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model: tf.keras.Model | None = None
        self.full_model: tf.keras.Model | None = None

    def get_base_model(self) -> tf.keras.Model:
        """Load the Xception base model"""
        self.model = tf.keras.applications.Xception(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        print(f"✅ Base Xception model loaded with weights={self.config.params_weights}")
        return self.model

    @staticmethod
    def _prepare_full_model(
        model: tf.keras.Model,
        classes: int,
        freeze_all: bool,
        freeze_till: int | None,
        learning_rate: float,
        dense_units: int,
        dropout_rate: float,
        weight_decay: float,
        optimizer_name: str,
        label_smoothing: float
    ) -> tf.keras.Model:
        """Freeze layers, add classifier head, compile model with regularization and LR scheduler"""

        # --- Freeze layers ---
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:freeze_till]:
                layer.trainable = False
            for layer in model.layers[freeze_till:]:
                layer.trainable = True

        # --- Classifier head ---
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        output_activation = "softmax" if classes > 2 else "sigmoid"
        predictions = tf.keras.layers.Dense(
            units=classes,
            activation=output_activation,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )(x)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=predictions)

        # --- Optimizer with LR scheduler ---
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )
        optimizer = (
            tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
            if optimizer_name.lower() == "adamw"
            else tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        )

        # --- Loss function ---
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

        # --- Compile model ---
        full_model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["accuracy"]
        )

        print("✅ Full model prepared (layers frozen, classifier head added, compiled with LR scheduler).")
        return full_model

    def update_base_model(self, fine_tune_at: int | None = None, fine_tune_last_n: int = 0) -> tf.keras.Model:
        """Create full model with classifier head and fine-tune last N layers"""
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=(fine_tune_at is None),
            freeze_till=fine_tune_at,
            learning_rate=self.config.params_learning_rate,
            dense_units=self.config.params_dense_units,
            dropout_rate=self.config.params_dropout_rate_head,
            weight_decay=self.config.params_weight_decay,
            optimizer_name=self.config.params_optimizer,
            label_smoothing=self.config.params_label_smoothing
        )

        # --- Fine-tune last N layers ---
        if fine_tune_last_n > 0:
            total_layers = len(self.full_model.layers)
            num_layers_to_unfreeze = min(fine_tune_last_n, total_layers)
            for layer in self.full_model.layers[-num_layers_to_unfreeze:]:
                layer.trainable = True
            print(f"✅ Last {num_layers_to_unfreeze} layers set to trainable for fine-tuning.")

        return self.full_model

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save the model in .h5 or .keras format"""
        path_str = str(path)
        if not (path_str.endswith(".h5") or path_str.endswith(".keras")):
            path_str += ".keras"
        model.save(path_str, include_optimizer=False)
        print(f"✅ Full model saved at: {path_str}")
