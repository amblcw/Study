import tensorflow as tf
tf.set_random_seed(777)

# data
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111,dtype=tf.float32)
b = tf.Variable(0,dtype=tf.float32)

# model
# y = wx + b <- 틀림 wx와 xw는 행렬이상에서 같지 않음
hypothesis = x*w + b

# compile
# model.compile(loss='mse',optimizer='SGD'와 같다
loss_fn = tf.reduce_mean(tf.square(hypothesis-y))   # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss_fn)

# fit 
sess = tf.compat.v1.Session()
init = tf.global_variables_initializer()
sess.run(init)

# model.fit
EPOCHS = 10000
for step in range(EPOCHS):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss_fn), sess.run(w), sess.run(b))    # model.weight
    
sess.close()