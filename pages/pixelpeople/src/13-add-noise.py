##### scales picture to 512x512 and adds noise

import cv2
import numpy as np
import argparse

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='noise',
                    description='adds noise')

    parser.add_argument('filename')
    args = parser.parse_args()

    image = cv2.imread(args.filename) # Only for grayscale image#
    dim = (64,64)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    rgba = cv2.cvtColor(resized, cv2.COLOR_RGB2RGBA)
    rgba[:,:,3] = 1

    # row,col,ch= rgba.shape
    # print(ch)

    # generator = torch.manual_seed(0)
    # torch_noise = torch.randn((row,col,ch),generator=generator).numpy()

    # noise = (torch_noise / 4 + 0.5).clip(0, 1)
    # noise = (noise*255).round()

    # noise = np.random.normal(128,128,(row,col,ch))
    # noise = noise.reshape(row,col,ch)

    # result = 0*rgba + 1*noise

    oname = "data/google.png"

    cv2.imwrite(oname, rgba)

    # test = cv2.imread(oname,cv2.IMREAD_UNCHANGED)
    # print(np.max(np.abs(result)))

    # #print(result)
    # #print(test)
    

    # print(np.max(np.abs(test-result))<0.01)

    # vec = test/255.0
    # vec = vec - 0.5
    # vec = 4*vec

    # print(torch_noise)
    # print(np.max(torch_noise))

    # print("hey")
    # print(vec)

    # print(np.max(np.abs(torch_noise-vec)))




