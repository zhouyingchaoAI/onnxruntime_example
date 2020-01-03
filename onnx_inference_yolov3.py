import onnxruntime
import numpy as np
import cv2
import time
from PIL import Image
INPUT_SIZE = 416

label = ["person",
        "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
        "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]
anchors = [[(116,90),(156,198),(373,326)],[(30,61),(62,45),(59,119)],[(10,13),(16,30),(33,23)]]

def sigmoid(x):
    s = 1 / (1 + np.exp(-1*x))
    return s
def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    image =  img[:,:,::-1].transpose((2,0,1))
    image = image[np.newaxis,:,:,:]/255
    image = np.array(image,dtype=np.float32)
    #返回原图像和处理后的数组
    return img,image


def process_image1(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    image_data = np.array(img, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return img, image_data

def getMaxClassScore(class_scores):
    class_score = 0
    class_index = 0
    for i in range(len(class_scores)):
        if class_scores[i] > class_score:
            class_index = i+1
            class_score = class_scores[i]
    return class_score,class_index

def getBBox(feat,anchors,image_shape,confidence_threshold):
    box = []
    for i in range(len(anchors)):
        for cx in range(feat.shape[0]):
            for cy in range(feat.shape[1]):
                tx = feat[cx][cy][0 + 85 * i]
                ty = feat[cx][cy][1 + 85 * i]
                tw = feat[cx][cy][2 + 85 * i]
                th = feat[cx][cy][3 + 85 * i]
                cf = feat[cx][cy][4 + 85 * i]
                cp = feat[cx][cy][5 + 85 * i:85 + 85 * i]

                bx = (sigmoid(tx) + cx)/feat.shape[0]
                by = (sigmoid(ty) + cy)/feat.shape[1]
                bw = anchors[i][0]*np.exp(tw)/image_shape[0]
                bh = anchors[i][1]*np.exp(th)/image_shape[1]

                b_confidence = sigmoid(cf)
                b_class_prob = sigmoid(cp)
                b_scores = b_confidence*b_class_prob
                b_class_score,b_class_index = getMaxClassScore(b_scores)

                if b_class_score > confidence_threshold:
                    box.append([bx,by,bw,bh,b_class_score,b_class_index])
    return box


def donms1(boxes,nms_threshold):
    b_x = boxes[:, 0]
    b_y = boxes[:, 1]
    b_w = boxes[:, 2] - b_x
    b_h = boxes[:, 3] - b_y
    scores = boxes[:,4]
    areas = (b_w+1)*(b_h+1)
    order = scores.argsort()[::-1]
    keep = []  # 保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i)  # 保留该类剩余box中得分最高的一个
        # 得到相交区域,左上及右下
        xx1 = np.maximum(b_x[i], b_x[order[1:]])
        yy1 = np.maximum(b_y[i], b_y[order[1:]])
        xx2 = np.minimum(b_x[i] + b_w[i], b_x[order[1:]] + b_w[order[1:]])
        yy2 = np.minimum(b_y[i] + b_h[i], b_y[order[1:]] + b_h[order[1:]])
        #相交面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #相并面积,面积1+面积2-相交面积
        union = areas[i] + areas[order[1:]] - inter
        # 计算IoU：交 /（面积1+面积2-交）
        IoU = inter / union
        # 保留IoU小于阈值的box
        inds = np.where(IoU <= nms_threshold)[0]
        order = order[inds + 1]  # 因为IoU数组的长度比order数组少一个,所以这里要将所有下标后移一位

    final_boxes = [boxes[i] for i in keep]
    return final_boxes

def donms(boxes,nms_threshold):
    b_x = boxes[:, 0]
    b_y = boxes[:, 1]
    b_w = boxes[:, 2]
    b_h = boxes[:, 3]
    scores = boxes[:,4]
    areas = (b_w+1)*(b_h+1)
    order = scores.argsort()[::-1]
    keep = []  # 保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i)  # 保留该类剩余box中得分最高的一个
        # 得到相交区域,左上及右下
        xx1 = np.maximum(b_x[i], b_x[order[1:]])
        yy1 = np.maximum(b_y[i], b_y[order[1:]])
        xx2 = np.minimum(b_x[i] + b_w[i], b_x[order[1:]] + b_w[order[1:]])
        yy2 = np.minimum(b_y[i] + b_h[i], b_y[order[1:]] + b_h[order[1:]])
        #相交面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #相并面积,面积1+面积2-相交面积
        union = areas[i] + areas[order[1:]] - inter
        # 计算IoU：交 /（面积1+面积2-交）
        IoU = inter / union
        # 保留IoU小于阈值的box
        inds = np.where(IoU <= nms_threshold)[0]
        order = order[inds + 1]  # 因为IoU数组的长度比order数组少一个,所以这里要将所有下标后移一位

    final_boxes = [boxes[i] for i in keep]
    return final_boxes


def drawBox(boxes,img):
    for box in boxes:
        x1 = int((box[0]-box[2]/2)*416)
        y1 = int((box[1]-box[3]/2)*416)
        x2 = int((box[0]+box[2]/2)*416)
        y2 = int((box[1]+box[3]/2)*416)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img, label[int(box[5])]+":"+str(round(box[4],3)), (x1+5,y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawBox1(boxes,img):
    for box in boxes:
        x1 = int(box[0]*img.shape[1])
        y1 = int(box[1]*img.shape[0])
        x2 = int((box[2])*img.shape[1])
        y2 = int((box[3])*img.shape[0])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img, label[int(box[5])]+":"+str(round(box[4],3)), (x1+5,y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawBox_yolov3(boxes, img):
        for box in boxes:
            x1 = int(box[1] * img.shape[1])
            y1 = int(box[0] * img.shape[0])
            x2 = int((box[3]) * img.shape[1])
            y2 = int((box[2]) * img.shape[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label[int(box[5])] + ":" + str(round(box[4], 3)), (x1 + 5, y1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def drawBox(boxes,img):
    for box in boxes:
        x1 = int(box[0]*img.shape[1])
        y1 = int(box[1]*img.shape[0])
        x2 = int((box[2])*img.shape[1])
        y2 = int((box[3])*img.shape[0])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img, label[int(box[5])]+":"+str(round(box[4],3)), (x1+5,y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getBoxes(prediction,confidence_threshold,nms_threshold):
    boxes = []
    for i in range(len(prediction)):
        feature_map = prediction[i][0].transpose((2, 1, 0))
        box = getBBox(feature_map, anchors[i], [INPUT_SIZE, INPUT_SIZE], confidence_threshold)
        boxes.extend(box)
    Boxes = donms(np.array(boxes),nms_threshold)
    return Boxes

def getBoxes1(prediction,confidence_threshold,nms_threshold):
    boxes = []
    for i in range(len(prediction)):
        box = prediction[i]
        if box[4] < confidence_threshold:
            continue
        boxes.append(box)
    Boxes = donms(np.array(boxes),nms_threshold)
    return Boxes

def TranPred(prediction):
    for i in range(len(prediction[0])):
        box = prediction[0][i]
        class_score = prediction[2][i]
        class_score = class_score[:, np.newaxis]
        class_id = prediction[1][i]
        class_id = class_id[:, np.newaxis]
        pred = np.concatenate((box, class_score, class_id), axis=1)

    return pred

def TranPred_yolov3(out_boxes, out_scores, out_classes):
    box = np.array(out_boxes)/416
    scores = np.expand_dims(np.array(out_scores), -1)
    classes = np.expand_dims(np.array(out_classes), -1)
    pred = np.concatenate((box, scores, classes), axis=1)

    return pred

def main():
    img, TestData = process_image1("timg.jpeg")
    session = onnxruntime.InferenceSession("yolov3.onnx")
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]

    ticks = time.time()

    print("inputs name:",inname, "outputs name:",outname)

    input = {
                inname[0]: TestData,
                inname[1]: np.expand_dims(np.array(TestData.shape[2:], dtype=np.float32), 0)
    }
    # print(TestData.shape)

    prediction = session.run(outname, input)
    print("const time %d", time.time()-ticks)
    # print(prediction)
    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in prediction[2]:
        out_classes.append(idx_[1])
        out_scores.append(prediction[1][tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(prediction[0][idx_1])


    boxes = TranPred_yolov3(out_boxes, out_scores, out_classes)
    drawBox_yolov3(boxes, img)

if __name__ == '__main__':
    main()