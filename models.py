from gcn.layers import *
from gcn.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None


        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.adjs = placeholders['support']
        self.num_support = len(self.adjs)
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        # self.loss += masked_bilinearsoftmax_cross_entropy(self.outputs, self.placeholders['labels'],
        #                                           self.placeholders['labels_mask'])

        self.loss += masked_bilinearsigmoid_cross_entropy(self.outputs, self.placeholders['labels'],
                                                          self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_bilinear_accuray(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        mixingWeights = tf.Variable(tf.random_normal([self.num_support, ]))
        self.mix = tf.nn.softmax(mixingWeights)
        adjs = []
        for aw in range(0, self.num_support):
            # adjs.append(tf.scalar_mul(mixingWeights[aw], tf.sparse_tensor_to_dense(self.adjs[aw])))
            adjs.append(tf.scalar_mul(self.mix[aw], self.adjs[aw]))
        mixedADJ = tf.add_n(adjs)
        self.mixedADJ = mixedADJ

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            support= mixedADJ,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=False,
                                            name="GCN1"))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            support=mixedADJ,
                                            act=tf.nn.softmax,
                                            dropout=True,
                                            logging=self.logging,
                                            name="GCN2"))


        # self.layers.append(Dense(input_dim=FLAGS.hidden2,
        #                          output_dim=self.input_dim,
        #                          placeholders=self.placeholders,
        #                          act=tf.nn.sigmoid,
        #                          dropout=True,
        #                          logging=self.logging))

        self.layers.append(BilinearOutput(input_dim=FLAGS.hidden2,
                                          output_dim=self.input_dim,
                                          placeholders=self.placeholders,
                                          act=lambda x:x,
                                          bias=False,
                                          linear=False,
                                          dropout=False,
                                          logging=self.logging,
                                          name="OutputLayer"))

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)


    def predict(self):
        return tf.nn.sigmoid(self.outputs)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_bilinearsigmoid_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_bilinear_accuray(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.sigmoid(self.outputs)


class SemiGraphEncoder(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(SemiGraphEncoder, self).__init__(**kwargs)

        self.embedding = None
        self.preds = None
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.x = placeholders['x_input']
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.adjs = placeholders['support']
        self.num_support = len(self.adjs)
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _accuracy(self):
        self.accuracy = masked_bilinear_accuray(self.preds, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        mixingWeights = tf.get_variable("Mixing", [self.num_support, ],
                                        initializer=tf.contrib.layers.xavier_initializer())
        #mixingWeights = tf.Variable(tf.random_normal([self.num_support, ]))
        self.mix = tf.nn.softmax(mixingWeights)
        adjs = []
        for aw in range(0, self.num_support):
            # adjs.append(tf.scalar_mul(mixingWeights[aw], tf.sparse_tensor_to_dense(self.adjs[aw])))
            adjs.append(tf.scalar_mul(self.mix[aw], self.adjs[aw]))
        mixedADJ = tf.add_n(adjs)
        self.mixedADJ = mixedADJ

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            support=mixedADJ,
                                            act=tf.nn.softmax,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=False,
                                            name="GCN1"))

        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
        #                                     output_dim=FLAGS.hidden2,
        #                                     placeholders=self.placeholders,
        #                                     support=mixedADJ,
        #                                     act=tf.nn.relu,
        #                                     dropout=True,
        #                                     logging=self.logging,
        #                                     name="GCN2"))
        #
        #
        #
        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
        #                                     output_dim=FLAGS.hidden1,
        #                                     placeholders=self.placeholders,
        #                                     support=mixedADJ,
        #                                     act=tf.nn.relu,
        #                                     dropout=True,
        #                                     logging=self.logging,
        #                                     name="GCN3"))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.input_dim,
                                            placeholders=self.placeholders,
                                            support=mixedADJ,
                                            act=tf.nn.sigmoid,
                                            dropout=True,
                                            logging=self.logging,
                                            name="GCN4"))

        self.predlayer = Dense(input_dim=FLAGS.hidden1,
                               output_dim=self.output_dim,
                               placeholders=self.placeholders,
                               act=lambda x:x,bias=True,name="Dense")


    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        self.embedding = self.activations[1]
        self.preds = self.predlayer(self.embedding)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def _loss(self):
        # Weight decay loss
        for layernum in range(len(self.layers)):
            for var in self.layers[layernum].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error

        self.loss += tf.reduce_mean(tf.squared_difference(self.outputs, self.x))

        self.loss += masked_bilinearsigmoid_cross_entropy(self.preds, self.placeholders['labels'],self.placeholders['labels_mask'])

    def predict(self):
        return tf.nn.sigmoid(self.preds)

class GraphEncoder(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GraphEncoder, self).__init__(**kwargs)

        self.embedding = None
        self.preds = None
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.x = placeholders['x_input']
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]#number of nodes
        self.adjs = placeholders['support']
        self.num_support = len(self.adjs)
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.x, self.placeholders['labels_mask'])

    # def normalize_adj(self, adj):
    #     adj = tf.eye(self.output_dim)+adj
    #     rowsum = tf.reduce_sum(adj,axis=1)
    #     d_inv_sqrt = tf.reciprocal(tf.sqrt(rowsum))
    #     d_mat_inv_sqrt = tf.diag(d_inv_sqrt)
    #     return tf.matmul(tf.matmul(d_mat_inv_sqrt,adj), d_mat_inv_sqrt)

    def _build(self):
        mixingWeights = tf.Variable(tf.random_normal([self.num_support,]))
        self.mix = tf.nn.softmax(mixingWeights)
        adjs = []
        for aw in range(0, self.num_support):
            #adjs.append(tf.scalar_mul(mixingWeights[aw], tf.sparse_tensor_to_dense(self.adjs[aw])))
            adjs.append(tf.scalar_mul(mixingWeights[aw], self.adjs[aw]))
        mixedADJ = tf.add_n(adjs)



        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            support = mixedADJ,
                                            placeholders=self.placeholders,
                                            act=tf.nn.softmax,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=False,
                                            name="GCN1"))

        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
        #                                     output_dim=FLAGS.hidden2,
        #                                     support=mixedADJ,
        #                                     placeholders=self.placeholders,
        #                                     act=tf.nn.sigmoid,
        #                                     dropout=True,
        #                                     logging=self.logging,
        #                                     name="GCN2"))
        #
        #
        #
        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
        #                                     output_dim=FLAGS.hidden1,
        #                                     support=mixedADJ,
        #                                     placeholders=self.placeholders,
        #                                     act=tf.nn.relu,
        #                                     dropout=True,
        #                                     logging=self.logging,
        #                                     name="GCN3"))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.input_dim,
                                            support=mixedADJ,
                                            placeholders=self.placeholders,
                                            act=tf.nn.sigmoid,
                                            dropout=True,
                                            logging=self.logging,
                                            name="GCN4"))


    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        self.embedding = self.activations[1]
        #self.preds = self.predlayer(self.embedding)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def _loss(self):
        # Weight decay loss
        for layernum in range(len(self.layers)):
            for var in self.layers[layernum].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error

        self.loss += masked_squreloss(self.outputs, self.x, self.placeholders['labels_mask'])


    def predict(self):
        return self.outputs
