import tensorflow as tf
def compute_saliency(model, image):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = prediction[:, 0]
    gradient = tape.gradient(loss, image)
    saliency = tf.reduce_max(tf.abs(gradient), axis=-1)
    return saliency.numpy()