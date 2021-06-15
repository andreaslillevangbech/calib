from utils.opticalflow import opticalFlowDense
import cv2
import os

def process_flows(src, dest):

    if not os.path.exists(dest):
        os.mkdir(dest)
    else:
        print('folder already exists')
        return

    template = 'flow_{0:05d}.jpg'

    paths = sorted([os.path.join(src, path) for path 
                            in os.listdir(src) if path.endswith('.jpg')])
    prev = cv2.imread(paths.pop(0))
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
    count = 0
    for path in paths:
        nxt = cv2.imread(path)
        nxt = cv2.cvtColor(nxt, cv2.COLOR_BGR2RGB)
        flow = opticalFlowDense(prev, nxt)
        cv2.imwrite(os.path.join(dest, template.format(count)), flow)
        prev = nxt
        count += 1

if __name__=="__main__":
    process_flows('labeled/0', 'labeled/flows0')
    print('all done')