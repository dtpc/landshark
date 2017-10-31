"""Train/test with tfrecords."""
import signal
import logging
import os.path
import pickle
from itertools import count

import numpy as np
import tensorflow as tf
import aboleth as ab
from sklearn.metrics import r2_score

log = logging.getLogger(__name__)
signal.signal(signal.SIGINT, signal.default_int_handler)


fdict = {
    "x_cat": tf.FixedLenFeature([], tf.string),
    "x_cat_mask": tf.FixedLenFeature([], tf.string),
    "x_ord": tf.FixedLenFeature([], tf.string),
    "x_ord_mask": tf.FixedLenFeature([], tf.string),
    "y": tf.FixedLenFeature([], tf.string)
    }


def train_data(records: list, batch_size: int, epochs: int=1) \
        -> tf.data.TFRecordDataset:
    dataset = tf.data.TFRecordDataset(records).repeat(count=epochs) \
        .shuffle(buffer_size=1000).batch(batch_size)
    return dataset


def test_data(records: list, batch_size: int, pred_samps: int=1) \
        -> tf.data.TFRecordDataset:
    """Train and test."""
    dataset = tf.data.TFRecordDataset(records).batch(batch_size).interleave(
        lambda x: tf.data.Dataset.from_tensors(x).repeat(pred_samps),
        cycle_length=1,
        block_length=pred_samps
        )
    return dataset


def decode(iterator, metadata):
    str_features = iterator.get_next()
    raw_features = tf.parse_example(str_features, features=fdict)
    npatch = (2 * metadata.halfwidth + 1) ** 2
    y_type = tf.float32 if metadata.target_dtype == np.float32 \
        else tf.int32
    with tf.name_scope("Inputs"):
        x_ord = tf.decode_raw(raw_features["x_ord"], tf.float32)
        x_cat = tf.decode_raw(raw_features["x_cat"], tf.int32)
        x_ord_mask = tf.decode_raw(raw_features["x_ord_mask"], tf.uint8)
        x_cat_mask = tf.decode_raw(raw_features["x_cat_mask"], tf.uint8)
        x_ord_mask = tf.cast(x_ord_mask, tf.bool)
        x_cat_mask = tf.cast(x_cat_mask, tf.bool)
        y = tf.decode_raw(raw_features["y"], y_type)

        x_ord.set_shape((None, npatch * metadata.nfeatures_ord))
        x_ord_mask.set_shape((None, npatch * metadata.nfeatures_ord))
        x_cat.set_shape((None, npatch * metadata.nfeatures_cat))
        x_cat_mask.set_shape((None, npatch * metadata.nfeatures_cat))
        y.set_shape((None, metadata.ntargets))

        # Placeholders for prediction
        xo_ = tf.placeholder_with_default(x_ord, x_ord.shape, name="Xo")
        xom_ = tf.placeholder_with_default(x_ord_mask, x_ord.shape, name="Xom")
        xc_ = tf.placeholder_with_default(x_cat, x_cat.shape, name="Xc")
        xcm_ = tf.placeholder_with_default(x_cat_mask, x_cat.shape, name="Xcm")
        y_ = tf.placeholder_with_default(y, y.shape, name="Y")

    return xo_, xom_, xc_, xcm_, y_


def load_metadata(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def predict_dict(d, Xo, Xom, Xc, Xcm):
    N = len(d.x_ord)
    xord = np.ma.reshape(d.x_ord, [N, -1])
    xcat = np.ma.reshape(d.x_cat, [N, -1])
    fdict = {Xo: xord.data, Xom: xord.mask, Xc: xcat.data, Xcm: xcat.mask}
    return fdict


def predict(model, metadata, data, n_samples):

    model_file = tf.train.latest_checkpoint(model)
    print("Loading model: {}".format(model_file))

    for i, d in enumerate(data):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            with sess.as_default():
                # TODO AL reloads/rewrites the graph in memory from protobuf
                # See glabrezu
                save = tf.train.import_meta_graph("{}.meta".format(model_file))
                save.restore(sess, model_file)

                # Restore place holders and prediction network
                Xo = graph.get_operation_by_name("Inputs/Xo").outputs[0]
                Xom = graph.get_operation_by_name("Inputs/Xom").outputs[0]
                Xc = graph.get_operation_by_name("Inputs/Xc").outputs[0]
                Xcm = graph.get_operation_by_name("Inputs/Xcm").outputs[0]
                phi = graph.get_operation_by_name("Deepnet/nnet").outputs[0]
                # TODO plus noise

                d_dict = predict_dict(d, Xo, Xom, Xc, Xcm)
                log.info("predicting batch {}".format(i))
                y_samples = ab.predict_samples(phi, d_dict, n_samples, sess)
                Ey = y_samples.mean(axis=0)
                Sf = y_samples.std(axis=0)
                yield Ey, Sf


def train_test(records_train, records_test, metadata, name, batch_size, epochs,
               n_samples, cf):

    train_dataset = train_data(records_train, batch_size, epochs)
    test_dataset = test_data(records_test, batch_size, n_samples)

    with tf.name_scope("Sources"):
        iterator = tf.data.Iterator.from_structure(
            train_dataset.output_types,
            train_dataset.output_shapes,
            shared_name="Iterator"
            )
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    Xo, Xom, Xc, Xcm, Y = decode(iterator, metadata)

    with tf.name_scope("Deepnet"):
        phi, lkhood, loss = cf.model(Xo, Xom, Xc, Xcm, Y, metadata)
        phi = tf.identity(phi, name="nnet")
        tf.summary.scalar("loss", loss)

    # Set up the trainig graph
    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer()
        global_step = tf.train.create_global_step()
        train = optimizer.minimize(loss, global_step=global_step)

    # Logging learning progress
    logger = tf.train.LoggingTensorHook(
        {"step": global_step, "loss": loss},
        every_n_iter=100
        )

    checkpoint_dir = os.path.join(os.getcwd(), name)
    r2 = -float("inf")

    # This is the main training "loop"
    with tf.train.MonitoredTrainingSession(
            config=cf.sess_config,
            checkpoint_dir=checkpoint_dir,
            scaffold=tf.train.Scaffold(local_init_op=train_init_op),
            save_summaries_steps=None,
            save_checkpoint_secs=20,
            save_summaries_secs=20,
            hooks=[logger]
            ) as sess:

        for i in count():
            log.info("Training round {} with {} epochs.".format(i, epochs))
            try:

                # Train loop
                sess.run(train_init_op)
                try:
                    while not sess.should_stop():
                        _, g = sess.run([train, global_step])
                except tf.errors.OutOfRangeError:
                    log.info("Training epoch complete.")

                # Test loop
                sess.run(test_init_op)
                Ys, EYs = [], []
                try:
                    while not sess.should_stop():
                        samples = []
                        for j in range(n_samples):
                            y, ey = sess.run([Y, phi])
                            samples.append(ey)

                        Ys.append(y)
                        cat_samples = np.concatenate(samples, axis=0)
                        EYs.append(cat_samples.mean(axis=0))
                except tf.errors.OutOfRangeError:
                    log.info("Testing epoch complete.")
                    pass

                # Scores
                Ys = np.vstack(Ys)
                EYs = np.vstack(EYs)
                r2 = r2_score(Ys, EYs, multioutput='raw_values')
                rsquare_summary(r2, sess, g)
                log.info("Aboleth r2: {}".format(r2))

            except KeyboardInterrupt:
                log.info("Training ended, final R-square = {}.".format(r2))
                break

    return checkpoint_dir


def rsquare_summary(r2, session, step=None):
    # Get a summary writer for R-square
    summary_writer = session._hooks[1]._summary_writer
    if np.isscalar(r2):
        sum_val = [tf.Summary.Value(tag='r-square', simple_value=r2)]
    else:
        sum_val = [tf.Summary.Value(tag='r-square-{}'.format(i),
                                    simple_value=r) for i, r in enumerate(r2)]
    score_sum = tf.Summary(value=sum_val)
    summary_writer.add_summary(score_sum, step)


def patch_slices(metadata):
    npatch = (metadata.halfwidth * 2 + 1) ** 2
    dim = npatch * metadata.nfeatures_cat
    begin = range(0, dim, npatch)
    end = range(npatch, dim + npatch, npatch)
    slices = [slice(b, e) for b, e in zip(begin, end)]
    return slices
