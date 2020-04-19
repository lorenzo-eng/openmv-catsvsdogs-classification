# TensorFlow Lite Cats vs Dogs Classification Example
#
# Classify in view dogs and cats.

import sensor, image, time, os, tf

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

# Load the built-in cats vs dogs classification network (the network is in the OpenMV Cam's firmware contained in this repo).
net = tf.load('catsvsdogs_classification')
labels = ['cat', 'dog']

clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot()

    for obj in net.classify(img, min_scale=1.0, scale_mul=0.99, x_overlap=0.0, y_overlap=0.0):
        print("**********\nClassification at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
        for i in range(len(obj.output())):
            print("%s = %f" % (labels[i], obj.output()[i]))
        img.draw_rectangle(obj.rect())
        img.draw_string(obj.x()+3, obj.y()-1, "%s: %d" % (labels[obj.output().index(max(obj.output()))], max(obj.output())*100), mono_space=False, scale=5)
    print(clock.fps(), "fps")

