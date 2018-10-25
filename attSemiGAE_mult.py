from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score,precision_recall_fscore_support

from gcn.utils import *
from gcn.models import GCN,SemiGraphEncoder
from gcn.attentionModel import AttSemiGraphEncoder

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'ddi', 'Dataset string.')
flags.DEFINE_string('model', 'attSemiGAE', 'Model string.')  # 'gcn', 'semiencoder', 'attSemiGAE'
flags.DEFINE_float('learning_rate', 0.05, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden0', 128, 'Number of units in hidden layer 0.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.') #dimension of the node embeddings
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-3, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


def ddi_load_data_GCN(dataset_str):
    """Load data."""
    names = ['allx', 'ally', 'graph',"adjmat", "trainMask", "valMask", "testMask"]
    objects = []
    for i in range(len(names)):
        with open("DDIdata/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, graphs, adjmats, train_mask, val_mask, test_mask = tuple(objects)
    tmpx = np.multiply(y, train_mask)
    features = x

    adjs = []
    for adjmat in adjmats:
        adjs.append(adjmat)

    return adjs, features, x, y, train_mask, val_mask, test_mask

# Load data
adjs, features, x, y, train_mask, val_mask, test_mask = ddi_load_data_GCN(FLAGS.dataset)

# print(sum(y[np.where(test_mask>0)]))

# Some preprocessing
#features = preprocess_features(densefeatures)
#features = sparse_to_tuple(densefeatures)
if FLAGS.model == 'gcn':
    support = []
    for adj in adjs:
        support.append(preprocess_adj_dense(adj))
    num_supports = len(support)
    model_func = GCN
elif FLAGS.model == 'semiencoder':
    support = []
    for adj in adjs:
        support.append(preprocess_adj_dense(adj))
    num_supports = len(support)
    model_func = SemiGraphEncoder
elif FLAGS.model =="attSemiGAE":
    support = []
    for adj in adjs:
        support.append(preprocess_adj_dense(adj))
    num_supports = len(support)
    model_func = AttSemiGraphEncoder
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=(None, features.shape[1])),
    'x_input': tf.placeholder(tf.float32, shape=(None, x.shape[1])),
    'labels': tf.placeholder(tf.float32, shape=(None, y.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

input_dim = features.shape[1]
# Create model
model = model_func(placeholders, input_dim=input_dim, logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(x, features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(x, features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
summary_writer = tf.train.SummaryWriter('./tmp/ddilogs', sess.graph)
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(x, features, support, y, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})


    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.activations,model.predict(),model.attention], feed_dict=feed_dict)
    mixedADJ = outs[-1]

    # attentions = outs[-1]
    # print(np.mean(attentions,axis=1))
    test_preds = outs[4]
    test_preds = test_preds + test_preds.T
    testsubs = np.where(test_mask > 0)
    roc = roc_auc_score(y[testsubs], test_preds[testsubs])
    prauc = average_precision_score(y[testsubs], test_preds[testsubs])
    print("Test set results:", roc, prauc)

    #summary_writer.add_summary(outs[0],epoch)

    # for var in tf.trainable_variables():
    #     sess.run(var)

    # Validation
    cost, acc,_, duration = evaluate(x, features, support, y, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
    #       "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
    #       "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        # print("Early stopping...")
        break

#print("Optimization Finished!")

# Testing
test_cost, test_acc, test_preds, test_duration = evaluate(x, features, support, y, test_mask, placeholders)
test_preds = test_preds+test_preds.T
testsubs = np.where(test_mask>0)
pkl.dump(mixedADJ,open("attention_Semi.pkl","wb"))
roc = roc_auc_score(y[testsubs], test_preds[testsubs])
prauc = average_precision_score(y[testsubs], test_preds[testsubs])

# print("Test set results:", "cost=", "{:.5f}".format(test_cost),
#       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
print("AttSemiGAE",roc, prauc)
# prec, recall, f, _ = precision_recall_fscore_support(y[testsubs]>=0.5, test_preds[testsubs]>=0.5, labels=[1,0])
# print(prec,recall,f)