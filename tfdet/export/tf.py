import os

import tensorflow as tf

from tfdet.core.util import save_model as save_tf, load_model as load_tf

def tf2lite(model, path, dtype = [tf.float32], optimizations = [tf.lite.Optimize.DEFAULT], data = None):
    if isinstance(model, tf.keras.Model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    elif isinstance(model, str):
        converter = tf.lite.TFLiteConverter.from_saved_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_concrete_functions(model)

    if optimizations is not None:
        if not isinstance(optimizations, list):
            optimizations = [optimizations]
        converter.optimizations = optimizations
    if data is None:
        if dtype is not None:
            if not isinstance(dtype, list):
                dtype = [dtype]
            converter.target_spec.supported_types = dtype
    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8 #tf.uint8
        converter.inference_output_type = tf.int8 #tf.uint8
        data = data if isinstance(data, list) or isinstance(data, tuple) else [data]
        def representative_dataset():
            for ds in zip(*data):
                yield [np.expand_dims(d, axis = 0) for d in ds]
        converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()
    
    name, ext = os.path.splitext(path)
    if len(ext) < 2:
        path = "{0}{1}".format(name, ".tflite")
    with open(path, "wb") as file:
        file.write(tflite_model)
    return path

def load_tflite(path, n_thread = None, content = None, predict = True):
    interpreter = tf.lite.Interpreter(path, model_content = content, num_threads = n_thread)
    if predict:
        signature_runner = interpreter.get_signature_runner()
        info = interpreter.get_signature_list()["serving_default"]
        input_keys = info["inputs"]
        def predict(*args, **kwargs):
            args = {k:v for k, v in zip(input_keys[:len(args)], args)}
            kwargs.update(args)
            pred = signature_runner(**{k:v if tf.is_tensor(v) else tf.convert_to_tensor(v) for k, v in kwargs.items()})
            return pred
        return predict
    else:
        return interpreter