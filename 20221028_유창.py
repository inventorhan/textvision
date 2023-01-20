import cv2
import numpy as np
import time
from glob import glob

def img_contrast(img): 
    # -----Converting image to LAB Color model----------------------------------- 
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3, 3)) 
    cl = clahe.apply(img) 
    # -----Merge the CLAHE enhanced L-channel with the a and b channel----------- 
    return cl

def contour_make(img, height, width, channel):
    
    contours, _= cv2.findContours(img,mode=cv2.RETR_EXTERNAL,
                                     method=cv2.CHAIN_APPROX_NONE)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, 
                    color=(255, 255, 255))

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    contours_dict = []
    num=0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), 
                    color=(255, 255, 255), thickness=2)
        
        contours_dict.append({
            'contour': contour,
            'id': num,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2),
            'xw':x+w,
            'yh':y+h
        })
        cv2.putText(temp_result,str(num),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
        num+=1
    return contours_dict, temp_result

def possible_contour_make(height, width, channel, contours_dict, MIN_AREA = 50, MIN_WIDTH = 5, MIN_HEIGHT = 5, MIN_RATIO = 0.15, MAX_RATIO = 1.4):
    
    possible_contours_dict = []
    contours_dict = sorted(contours_dict, key = lambda x : x['x'])
    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
        
        if area > MIN_AREA \
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours_dict.append(d)
            
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours_dict:
        cv2.putText(temp_result,str(d['idx']),(d['x'],d['y']),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), 
                    color=(255, 255, 255), thickness=2)
        cv2.drawContours(temp_result, d['contour'],-1,(0,255,0),3)
    
    return possible_contours_dict, temp_result

def img_align_work(img, height, width, possible_contours_dict, PLATE_WIDTH_PADDING = 1.3, PLATE_HEIGHT_PADDING = 2.5, MIN_PLATE_RATIO = 3, MAX_PLATE_RATIO = 30):

    plate_imgs = []
    plate_infos_dict = []

    sorted_chars = sorted(possible_contours_dict, key=lambda x: x['cx'])

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )

    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

    img_rotated = cv2.warpAffine(img, M=rotation_matrix, dsize=(width, height))

    img_cropped = cv2.getRectSubPix(
        img_rotated, 
        patchSize=(int(plate_width), int(plate_height)), 
        center=(int(plate_cx), int(plate_cy))
    )

    # if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        # continue

    plate_infos_dict.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })
    
    return plate_infos_dict, img_cropped

def text_word_separate(img, height, width, channel, contours_dict):
    word_img_list = []
    if len(contours_dict) == 0:
        temp_result = np.zeros((height, width), dtype=np.uint8)
        return word_img_list, temp_result
    # height max
    axis_y = min(contours_dict, key = lambda x:x['y'])['y']
    axis_yh_max = max(contours_dict, key = lambda x:x['yh'])['yh']-axis_y
    # width max
    axis_x = min(contours_dict, key = lambda x:x['x'])['x']
    axis_xw_max = max(contours_dict, key = lambda x:x['xw'])['xw']-axis_x
    
    temp_result = np.zeros((axis_yh_max, axis_xw_max), dtype=np.uint8)
    
    for contour in contours_dict:
        # print(contour['y'],contour['h'], contour['x'],contour['w'])
        dst = img[contour['y']:contour['y']+contour['h'], contour['x']:contour['x']+contour['w']]
        temp_result[contour['y']-axis_y:contour['y']-axis_y+contour['h'], contour['x']-axis_x:contour['x']-axis_x+contour['w']] = dst
        word_img_list.append(dst)
        
    return word_img_list, temp_result

def img_pretreatment(img_ori):
    
    height, width, channel = img_ori.shape
    
    #bgr to gray
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    
    # 03 Activate threshold image
    kernel = np.ones((5,5),np.uint8)
    # img_erode = cv2.erode(gray, kernel, iterations=1)
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    img_thresh = cv2.adaptiveThreshold(
        img_blurred, 
        maxValue=255, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=19, 
        C=9
        )

    img_dilate = cv2.dilate(img_thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    
    try:
        # text align
        contours_dict, contours_dict_img = contour_make(img_thresh.copy(), height, width, channel)
        possible_contours_dict, possible_contour_img = possible_contour_make(height, width, channel, contours_dict)
        plate_infos_dict, img_cropped = img_align_work(img_thresh.copy(), height, width, possible_contours_dict)
        
        # text contour
        contours_dict2, contours_dict_img2 = contour_make(img_cropped.copy(), height, width, channel)
        possible_contours_dict2, possible_contour_img2 = possible_contour_make(height, width, channel, contours_dict2)
        result_img_list, result_img = text_word_separate(img_cropped.copy(), height, width, channel, possible_contours_dict2)
        # result_img = possible_contour_img2
    except:
        result_img_list = []
        result_img = np.zeros((height, width), dtype=np.uint8)
        
    return result_img_list, result_img
        

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while (1):
        k = cv2.waitKey(100)                
        if k == ord('q'):
            break
        elif k == ord('p'):
            pass
        elif k == ord('s'):
            for i, img in enumerate(result_img_list):
                temp = str(i)
                cv2.imwrite(f"./image/train/{temp}.jpg",img)
        else:
            ret, frame = capture.read()
        
        # dst = frame[200:350, 50:600]
        result_img_list, result_img = img_pretreatment(frame)
        cv2.imshow("VideoFrame", frame)
        cv2.imshow("VideoResult", result_img)

    capture.release()
    cv2.destroyAllWindows()
    
    
    # filepath = "./image/roiset.png"
    # img_ori = cv2.imread(filepath, cv2.IMREAD_COLOR)


    # result_img_list, result_img = img_pretreatment(img_ori)
    # # result_img_list = img_pretreatment(img_ori)
    # # for i, img in enumerate(result_img_list):
    # #     cv2.imshow(str(i),img)
    # cv2.imshow('result image', result_img)
    # cv2.waitKey(0)