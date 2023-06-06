from PIL import Image
import PIL
import glob

def process():
    images = []
    labels = []
    for i in glob.glob('uploads/*'):
        ind = i.split('.')
        images.append(i)
        labels.append(ind[0][-1])



if __name__ == "__main__":
    process()