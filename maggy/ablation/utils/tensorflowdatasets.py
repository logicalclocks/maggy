#
#   Copyright 2020 Logical Clocks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#


def ablate_feature_and_create_tfrecord_dataset_from_featurestore(
    ablated_feature,
    training_dataset_name,
    training_dataset_version,
    label_name,
    num_epochs,
    batch_size,
    shuffle_buffer_size=10000,
):

    import tensorflow as tf
    from hops import featurestore

    dataset_dir = featurestore.get_training_dataset_path(
        training_dataset_name, training_dataset_version
    )
    input_files = tf.gfile.Glob(dataset_dir + "/part-r-*")
    dataset = tf.data.TFRecordDataset(input_files)
    tf_record_schema = featurestore.get_training_dataset_tf_record_schema(
        training_dataset_name
    )
    meta = featurestore.get_featurestore_metadata()
    training_features = [
        feature.name
        for feature in meta.training_datasets[
            training_dataset_name + "_" + str(training_dataset_version)
        ].features
    ]

    if ablated_feature is not None:
        training_features.remove(ablated_feature)

    training_features.remove(label_name)

    def decode(example_proto):
        example = tf.parse_single_example(example_proto, tf_record_schema)
        # prepare the features
        x = []
        for feature_name in training_features:
            # temporary fix for the case of tf.int types
            if tf_record_schema[feature_name].dtype.is_integer:
                x.append(tf.cast(example[feature_name], tf.float32))
            else:
                x.append(example[feature_name])

        # prepare the labels
        if tf_record_schema[label_name].dtype.is_integer:
            y = [tf.cast(example[label_name], tf.float32)]
        else:
            y = [example[label_name]]

        return x, y

    dataset = (
        dataset.map(decode)
        .shuffle(shuffle_buffer_size)
        .batch(batch_size)
        .repeat(num_epochs)
    )
    return dataset
