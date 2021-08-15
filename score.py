import joblib
import json
import numpy as np
import os

#from inference_schema.schema_decorators import input_schema, output_schema
#from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'nb_model.pkl')
    # Deserialize the model file back into a sklearn model.
    model = joblib.load(model_path)

#input_sample = np.array([[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
#output_sample = np.array([3726.995])

#@input_schema('data', NumpyParameterType(input_sample))
#@output_schema(NumpyParameterType(output_sample))
def run(data):
    data = json.loads(data)["data"]
    try:
        result = model.predict(data)
        # You can return any JSON-serializable object.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error