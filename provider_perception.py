import tensorflow as tf

def read(file_names, num_epochs=1, batch_size=16, num_points=1024):
    feature = {'points': tf.FixedLenFeature([], tf.string),
               'distance': tf.FixedLenFeature([], tf.float32),
               'closest/r': tf.FixedLenFeature([], tf.float32)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(file_names, num_epochs=num_epochs)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    points = tf.decode_raw(features['points'], tf.float32)
    
    # Cast label data into int32
    distance = tf.cast(features['closest/r'], tf.float32)
    M = 23.143643224367651
    S = 12.196518821222815
    distance = (distance - M) / S
    # Reshape image data into the original shape
    points = tf.reshape(points, [num_points, 3])
    points = points / S
    
    # Any preprocessing here ...
    
    # Creates batches by randomly shuffling tensors
    return tf.train.shuffle_batch(
                [points, distance],
                batch_size=batch_size, capacity=30, num_threads=1, 
                min_after_dequeue=10)
