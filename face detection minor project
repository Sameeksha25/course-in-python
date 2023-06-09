import cv2
#load cascade classifier 
face_cap=cv2.CascadeClassifier("C:/Users/new/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

#to capture video 
video_cap=cv2.VideoCapture(0)
while True:
    #reads the frame
    ret , video_d = video_cap.read()
    #converts it to grayscale image
    col = cv2.cvtColor(video_d,cv2.COLOR_BGR2GRAY)
    #detects the face 
    faces=face_cap.detectMultiScale(
    col,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE
    )
    #draw rectangle around the face
    for(x,y,w,h) in faces:
        cv2.rectangle(video_d,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("live_video",video_d)
    #stop incase exit key pressed
    if cv2.waitKey(10)==ord("a"):
        break
video_cap.release()
