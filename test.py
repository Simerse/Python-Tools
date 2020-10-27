
import simerse.image_util
import imageio
import csv

im = imageio.imread('C:/Users/hauck/Pictures/Simerse/Tutorial/Unreal 4/Keypoints.png')
encoded, uid_mapping = simerse.image_util.to_keypoint(im)

imageio.imwrite('C:/Users/hauck/Pictures/Simerse/Tutorial/Unreal 4/KeypointsEncoded.png', encoded)

with open('C:/Users/hauck/Documents/test.csv', 'w', newline='') as f:
    csv.writer(f).writerows(uid_mapping)
