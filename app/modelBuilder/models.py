

from typing import Dict, Text

import tensorflow as tf
import tf_keras as ks

class NoBaseClassMovielensModel(ks.Model):
    def __init__(self, user_model, movie_model,task):
        super().__init__()
        self.movie_model = movie_model
        self.user_model = user_model
        self.task = task

    def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        with tf.GradientTape() as tape:
            user_embeddings = self.user_model(features["user_id"])
            positive_movie_embeddings = self.movie_model(features["item_id"])
            loss = self.task(user_embeddings, positive_movie_embeddings)
            regularization_loss = sum(self.losses)
            total_loss = loss + regularization_loss
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss, "regularization_loss": regularization_loss, "total_loss": total_loss}