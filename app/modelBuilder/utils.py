from typing import Dict, Text

import tensorflow as tf
import tf_keras as ks


class BaseTopKLayer(ks.layers.Layer):
    def __init__(self, user_model, k=10, name="base_top_k_layer", **kwargs):
        super().__init__(name=name, **kwargs)
        self.user_model = user_model
        self.k = k
        self.candidate_embeddings = None
        self.candidate_titles = None

    def index_from_dataset(self, dataset):
        candidate_titles = []
        candidate_embeddings = []

        for titles, embeddings in dataset:
            candidate_titles.append(titles)
            candidate_embeddings.append(embeddings)

        self.candidate_titles = tf.concat(candidate_titles, axis=0)
        self.candidate_embeddings = tf.concat(candidate_embeddings, axis=0)

    def compute_similarity(self, user_embeddings):
        if self.candidate_embeddings is None or self.candidate_titles is None:
            raise ValueError("Candidates must be indexed before querying.")
        return tf.linalg.matmul(user_embeddings, self.candidate_embeddings, transpose_b=True)

    def call(self, user_ids):
        user_embeddings = self.user_model(user_ids)
        similarity_scores = self.compute_similarity(user_embeddings)
        top_k_scores, top_k_indices = tf.math.top_k(similarity_scores, k=self.k)
        top_k_titles = tf.gather(self.candidate_titles, top_k_indices)
        return top_k_scores, top_k_titles

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def retrieve(self, user_ids):
        return self.call(user_ids)


class CustomFactorizedTopK(BaseTopKLayer):
    def __init__(self, user_model, k=10, name="custom_factorized_top_k", **kwargs):
        super().__init__(user_model, k, name, **kwargs)


class CustomRetrievalTask(ks.layers.Layer):

    def __init__(self, custom_metrics=None, name="custom_retrieval_task", **kwargs):
        super().__init__(name=name, **kwargs)
        self.custom_metrics = custom_metrics if custom_metrics else []

    def call(self, user_embeddings, candidate_embeddings, labels=None):

        logits = tf.linalg.matmul(user_embeddings, candidate_embeddings, transpose_b=True)

        labels_one_hot = (
            tf.one_hot(labels, depth=tf.shape(candidate_embeddings)[0])
            if labels is not None
            else tf.eye(tf.shape(candidate_embeddings)[0])
        )

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
        )

        for metric in self.custom_metrics:
            metric.update_state(labels_one_hot, tf.nn.softmax(logits))

        return loss

    def compute_metrics(self):

        return {metric.name: metric.result().numpy() for metric in self.custom_metrics}

    def reset_metrics(self):

        for metric in self.custom_metrics:
            metric.reset_states()


class CustomCallback(ks.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} ended.")
        print(f"Training loss: {logs['loss']}, Validation loss: {logs['val_loss']}")
        print(f"Training accuracy: {logs['accuracy']}, Validation accuracy: {logs['val_accuracy']}")