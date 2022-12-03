import cv2
import numpy as np
def peer2(img):
        b, g, r = cv2.split(img)
        ret, m1 = cv2.threshold(r, 95, 255, cv2.THRESH_BINARY)
        ret, m2 = cv2.threshold(g, 30, 255, cv2.THRESH_BINARY)
        ret, m3 = cv2.threshold(b, 20, 255, cv2.THRESH_BINARY)
        mmax = cv2.max(r, cv2.max(g, b))
        mmin = cv2.min(r, cv2.min(g, b))

        ret, m4 = cv2.threshold(mmax - mmin, 15, 255, cv2.THRESH_BINARY)
        ret, m5 = cv2.threshold(cv2.absdiff(r, g), 15, 255, cv2.THRESH_BINARY)
        m6 = cv2.compare(r, g, cv2.CMP_GE)
        m7 = cv2.compare(r, b, cv2.CMP_GE)
        mask = m1 & m2 & m3 & m6 & m4 & m5 & m7

        return mask

    #_, img = vc.read()
    # input = np.array([cv2.resize(img,(128,128))],np.float32)/255
    # y = model.predict(input)
    # # print y[0]
    # ans = np.argmax(y[0])
    # print labels['class'][ans]
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img,labels['class'][ans],(10,400), font, 4,(255,255,255),2,cv2.LINE_AA)
#img = cv2.imread("/Users/shreenidhir/Documents/Machine learning/project_18nov/rps_data/scissors/2l1K148aIJHRR1q7.png")
img=cv2.imread("/Users/shreenidhir/Documents/Machine learning/project_18nov/contour_folder/contour_input/G6trRFSUGIeaQorS.png")
th3 = peer2(img)
kernel = np.ones((7,7), np.uint8)
closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
# gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)
contours, hiearachy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt = contours[max_index]
(x,y,w,h) = cv2.boundingRect(cnt)
print("x==",x)
print("y==",y)
print("w==",w)
print("h==",h)
start_point=(x,y)
end_point=(x+w,y+h)
color = (255, 0, 0)
thickness = 2
frame = cv2.rectangle(img, start_point, end_point, color, thickness)
cv2.imwrite("/Users/shreenidhir/Documents/Machine learning/project_18nov/shree_new_contour_img_centre.jpg",img)
#print(contours)
cv2.drawContours(img, contours, max_index, (255, 255, 255), 3)
hull = cv2.convexHull(cnt, returnPoints=False)
# cv2.drawContours(img,[hull],0,(0,255,0),3)
defects = cv2.convexityDefects(cnt,hull)
print(defects.shape)
cv2.imwrite("/Users/shreenidhir/Documents/Machine learning/project_18nov/shree_new_contour1.jpg",img)