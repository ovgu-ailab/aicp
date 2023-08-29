##### scales picture to 512x512 and adds noise

import cv2
import numpy as np
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='noise',
                    description='adds noise')

    parser.add_argument('filename')
    args = parser.parse_args()

    image = cv2.imread(args.filename) # Only for grayscale image#
    dim = (512,512)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    row,col,ch= resized.shape
    noise = np.random.normal(0,500,(row,col,ch))
    noise = noise.reshape(row,col,ch)

    result = resized + noise

    cv2.imwrite(args.filename+"_noisy", result)




