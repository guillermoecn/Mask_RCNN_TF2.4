import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import random
import matplotlib.pyplot as plt
import time
import numpy as np

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG', 'banda', 'glomero', 'roca'] # Deben coincidir con el orden de id del json
# CLASS_NAMES = ['BG', 'stone']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

model_path = "./models/rocks_mrcnn_cobredelmayo_dataset_augmented_20ep.h5"
# Load the weights into the model.
model.load_weights(filepath=model_path, by_name=True)

# load the input image, convert it from BGR to RGB channel
# /content/Mask_RCNN/kangaroo-transfer-learning/kangaroo/images/00001.jpg
imgs_path = './datasets/rocks-test3-augmented/test/'
imgs_list = os.listdir(imgs_path)
random.shuffle(imgs_list)
imgs_paths = [os.path.join(imgs_path, file) for file in imgs_list if file.split(".")[-1] != "json"]
# image_path = os.path.join(imgs_path, random.choice(os.listdir(imgs_path)))
times = []
for i, image_path in enumerate(imgs_paths):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform a forward pass of the network to obtain the results
    print("Detection")
    t1 = time.time()
    r = model.detect([image], verbose=0)
    t2 = time.time()
    print("Inference time:", (t2-t1))
    print("Image:", image_path)

    times.append((t2-t1))

    # Get the results for the first image.
    r = r[0]

    plt.imshow(image)
    # Visualize the detected objects.
    res = mrcnn.visualize.display_instances(image=image, 
                                    boxes=r['rois'], 
                                    masks=r['masks'], 
                                    class_ids=r['class_ids'], 
                                    class_names=CLASS_NAMES, 
                                    scores=r['scores'])

    # plt.imsave(f"./test_imgs/inference_{i}_original.jpg", image)
    # plt.imsave(f"./test_imgs/inference_{i}_prediction.jpg", res)
    # res.savefig(f"./test_imgs/inference_{i}_prediction.jpg")

    print(f'Masks: {r["masks"].shape}')
# plt.imsave("result1.jpg", r["masks"][:,:,0])
# plt.imsave("result2.jpg", r["masks"][:,:,1])
# plt.imsave("result3.jpg", r["masks"][:,:,2])

print("Inference average time:", np.mean(times[1:]))






