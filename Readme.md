## Object Classification with TFLite Micro using the OpenMV Cam H7

This repository contains the code necessary to train and run a *cats vs dogs* classification model on the [OpenMV Cam H7](https://openmv.io).
It should be possible to generalize it to classify arbitrary classes of objects.

It is suggested to execute the scripts contained in the `model_prepare` folder within a Docker container built through the `Dockerfile` which is found in this folder.

### Build Docker Image

    docker build -t embon/tf .

### Run Docker Image


    UID_GID=$(id -u):$(id -g) docker-compose run embontf

`UID_GID` variable needed to create a non-root user inside the container to run tf scripts in a safer way.

**n.b** if you run docker with root user, but your non-root user is able to gain root privileges using the `sudo` command, run:

    UID_GID=$(id -u):$(id -g) sudo -E docker-compose run embontf

To preserve environment variables.

The container is started sharing `model_prepare` folder with the host (useful to easily get training output files on the host).

### Train the model and compile the OpenMV Firmware

Train the model:

    python model_train.py

Outputs the model to `build/keras`, and `build/training_out.png` image of the plots of accuracy and loss for training and validation.
Convert to TFLite format (output in `build`):

    python model_convert.py

Outputs converted model to `build/catsvsdogs_model.tflite`.
Test TFLite model:

    python model_test.py

Outputs `build/test_out.png` image containing pictures labeled during the test.
Convert TFLite model to c src and compile archive to include within OpenMV firmware:

    python model_carch.py

Outputs `build/libtf_catsvsdogs_classify_model_data.a` archive.
It is possible to change some parameters of this process inside the `config.py` file.

### Recompile OpenMV firmware

    ./openmv_download_build.sh

### Run Cats vs Dogs Classification example

1. exit the Docker container
2. open OpenMV IDE
3. `Tools->Run Bootloader` to flash the OpenMV firmware with the `cats vs dogs` model built-in (the one you compiled following the previous steps or the `firmware.dfu` file you find in this folder)
4. open `tf_catsvsdogs_classification_search_whole_window.py` script and run it
5. put a cat or a dog in front of the camera
