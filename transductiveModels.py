from gcn.layers import *
from gcn.metrics import *
from models import Model

flags = tf.app.flags
FLAGS = flags.FLAGS


class TransductiveGraphEncoder(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(TransductiveGraphEncoder, self).__init__(**kwargs)

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
        self.accuracy = masked_accuracy(self.outputs, self.x, self.placeholders['labels_mask'])

    def _build(self):
        #mixingWeights = tf.Variable(tf.random_normal([self.num_support, ]))
        self.mixingWeights = tf.get_variable("Mixing",[self.num_support, ],initializer=tf.contrib.layers.xavier_initializer())
        self.mix = tf.nn.softmax(self.mixingWeights)
        adjs = []
        for aw in range(0, self.num_support):
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
        #                                     act=tf.nn.softmax,
        #                                     dropout=True,
        #                                     logging=self.logging,
        #                                     name="GCN2"))
        #
        #
        #
        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
        #                                     output_dim=FLAGS.hidden1,
        #                                     placeholders=self.placeholders,
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


    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()


        #trainMask = tf.cast(self.placeholders['labels_mask'],dtype="float32")
        supLabels = tf.Variable(tf.zeros_initializer((self.output_dim, self.output_dim)))

        inputLabels = tf.add(self.inputs, supLabels)
        self.supLabels = supLabels

        # Build sequential layer model
        self.activations.append(inputLabels)
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
        self.loss += masked_squreloss(self.supLabels, 0, self.placeholders['labels_mask'])
        #self.loss += tf.reduce_mean(tf.squared_difference(self.supLabels,0))
        self.loss += masked_squreloss(self.outputs, self.supLabels, self.placeholders['test_mask'])


    def predict(self):
        return self.outputs
