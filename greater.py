import tensorflow as tf 
import random

if __name__ == '__main__':
  num_input = 2
  num_h1 = 16
  num_h2 = 8
  num_output = 1
  input_layer = tf.placeholder(shape=[None, num_input], dtype=tf.float32)
  output_layer = tf.placeholder(shape=[None, num_output], dtype=tf.float32)

  w1 = tf.Variable(tf.random_normal(shape=[num_input, num_h1], stddev=10))
  b1 = tf.Variable(tf.random_normal(shape=[num_h1], stddev=10))
  h1 = tf.matmul(input_layer, w1) + b1 

  w2 = tf.Variable(tf.random_normal(shape=[num_h1, num_h2], stddev=10))
  b2 = tf.Variable(tf.random_normal(shape=[num_h2], stddev=10))
  h2 = tf.matmul(h1, w2) + b2

  w3 = tf.Variable(tf.random_normal(shape=[num_h2, num_output], stddev=10))
  b3 = tf.Variable(tf.random_normal(shape=[num_output], stddev=10))
  model_output = tf.matmul(h2, w3) + b3

  xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits = model_output, labels = output_layer))

  opt = tf.train.AdamOptimizer(0.001)
  train_step = opt.minimize(xentropy)

  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)

  prediction = tf.round(tf.sigmoid(model_output))
  predictions_correct = tf.cast(tf.equal(prediction, output_layer), tf.float32)
  reward = tf.reduce_mean(predictions_correct)

  for i in range(2000):
    x1 = random.random() * 10000
    x2 = random.random() * 10000
    y = 1 if x1 > x2 else 0

    sess.run(train_step, feed_dict = {input_layer: [[x1, x2]], output_layer: [[y]]})


  total_reward = 0
  for i in range(1000):
    x1 = random.random() * 10000 + 10000
    x2 = random.random() * 10000 + 10000
    y = 1 if x1 > x2 else 0

    total_reward += sess.run(reward, feed_dict = {input_layer: [[x1, x2]], output_layer: [[y]]})

  print(total_reward)
