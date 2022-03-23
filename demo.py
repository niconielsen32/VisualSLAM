#!/usr/bin/env python3
import glob

import cv2
import numpy as np

import dbow


np.random.seed(123)

print("Loading Images")
images_path = glob.glob("./images/*.png")
images = []
for image_path in images_path:
    images.append(cv2.imread(image_path))

n_clusters = 10
depth = 2
print("Creating Vocabulary")
vocabulary = dbow.Vocabulary(images, n_clusters, depth)

orb = cv2.ORB_create()

print("Creating Bag of Binary Words from Images")
bows = []
for image in images:
    kps, descs = orb.detectAndCompute(image, None)
    descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
    bows.append(vocabulary.descs_to_bow(descs))

print("Creating Database")
db = dbow.Database(vocabulary)
for image in images:
    kps, descs = orb.detectAndCompute(image, None)
    descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
    db.add(descs)

print("Querying Database")
for image in images:
    kps, descs = orb.detectAndCompute(image, None)
    descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
    scores = db.query(descs)
    match_bow = db[np.argmax(scores)]
    match_desc = db.descriptors[np.argmax(scores)]

print("Saving and Loading Vocabulary")
vocabulary.save("vocabulary.pickle")
loaded_vocabulary = vocabulary.load("vocabulary.pickle")
for image in images:
    kps, descs = orb.detectAndCompute(image, None)
    descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
    loaded_vocabulary.descs_to_bow(descs)

print("Saving and Loading Database")
db.save("database.pickle")
loaded_db = db.load("database.pickle")
for image in images:
    kps, descs = orb.detectAndCompute(image, None)
    descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
    scores = loaded_db.query(descs)
