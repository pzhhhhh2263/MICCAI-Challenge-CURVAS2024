"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path
from glob import glob
import SimpleITK
import torch
import numpy as np
import gc, os
import time
from resources.nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from resources.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subfiles, join
import threading
import SimpleITK as sitk


print('All required modules are loaded!!!')
# INPUT_PATH = Path("/input")
# OUTPUT_PATH = Path("/output")
# RESOURCE_PATH = Path("resources")

INPUT_PATH = Path("./test/input")
OUTPUT_PATH = Path("./test/output")
RESOURCE_PATH = Path("resources")

nnUNet_raw = INPUT_PATH / "images/thoracic-abdominal-ct"
nnUNet_source = str(RESOURCE_PATH)


def perform_inference(input_image):    
    # Define nnUNet v2 model parameters
    model_name = '3d_fullres'
    trainer_class_name = 'nnUNetTrainer_NoMirroring_ep500'
    plans_identifier = 'nnUNetResEncUNetPlans_24G'
    task_name = 'Dataset511_CURVAS'
    # Create nnUNet predictor        
    predictor = nnUNetPredictor(
            tile_step_size=1.0,
            use_gaussian=True,
            use_mirroring=False,
            device=torch.device('cuda'),
            perform_everything_on_device=True,
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=False
        )

    model_path = os.path.join(nnUNet_source, task_name, f"{trainer_class_name}__{plans_identifier}__{model_name}")
    print('Model path: '+str(model_path))

    predictor.initialize_from_trained_model_folder(
            model_path,
            use_folds=("all"),
            checkpoint_name="checkpoint_best.pth",
    )

    print(input_image)

    image, properties = SimpleITKIO().read_images([input_image])
    # ret = predictor.predict_single_npy_array(image, properties, None, None, False)
    iterator = predictor.get_data_iterator_from_raw_npy_data([image], None, [properties], None, 4)
    # output_segmentation, probabilities = predictor.predict_from_data_iterator(iterator, True, 3)
    result = predictor.predict_from_data_iterator(iterator, True, 4)

    # result[0][0]相当于output_segmentation， result[0][1]probabilities
    output_segmentation = result[0][0]
    probabilities1 = result[0][1][1]
    probabilities2 = result[0][1][2]
    probabilities3 = result[0][1][3]

    # To improve write speeds.
    # probabilities1[probabilities1 < 0.0001] = 0
    # probabilities2[probabilities2 < 0.0001] = 0
    # probabilities3[probabilities3 < 0.0001] = 0

    # -6  2.10s
    # -5  1.02s
    # -4  1.02s
    # nouse
    print('Prediction finished')

    return output_segmentation, probabilities1,  probabilities2, probabilities3


def write_files_in_parallel(files_data):
    threads = []
    for location, array in files_data:
        print('location: '+str(location))
        thread = threading.Thread(target=write_array_as_image_file, kwargs={'location': location, 'array': array})
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    suffix = ".mha"

    image = sitk.GetImageFromArray(array)
    sitk.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )

def run():
    _show_torch_cuda_info()
    start_time = time.time()

    # Read the input
    os.environ['nnUNet_compile'] = 'F' 
    ct_mha = subfiles(nnUNet_raw, suffix='.mha')[0]

    output_abdominal_organ_segmentation, output_pancreas_confidence, output_kidney_confidence, output_liver_confidence = perform_inference(
        input_image=ct_mha
    )


    prediction_time = np.round((time.time() - start_time)/60, 3)
    print('Prediction time: '+str(prediction_time))
    print('Saving the predictions')

    start_time = time.time()

    write_files_in_parallel([(Path(OUTPUT_PATH / "images/abdominal-organ-segmentation"), output_abdominal_organ_segmentation),
                            (Path(OUTPUT_PATH / "images/kidney-confidence"), output_kidney_confidence),
                            (Path(OUTPUT_PATH / "images/pancreas-confidence"), output_pancreas_confidence),
                            (Path(OUTPUT_PATH / "images/liver-confidence"), output_liver_confidence)
                            ])
    

    saving_time = np.round((time.time() - start_time)/60, 3)
    print('Saving time: '+str(saving_time))
    print('Finished running algorithm!')
    print('Saved!!!')
    return 0


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())



