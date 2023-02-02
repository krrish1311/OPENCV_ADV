import cv2
import mediapipe as mp
import time



class handdetector :
    def __init__(self,mode=False ,max_hands=2,detconf=0.5,trackconf=0.5):
        self.mode=mode
        self.max_hands=max_hands
        self.detconf=detconf
        self.trackconf=trackconf
        self.mphands=mp.solutions.hands
        self.hands=self.mphands.Hands(self.mode,self.max_hands,self.detconf,self.trackconf)
        self.mpdraw=mp.solutions.drawing_utils

    def find_hands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img,handlms,self.mphands.HAND_CONNECTIONS)

        return img


    def find_positions(self,img,hand_no=0,draw=True):
        lmlist=[]
        if self.results.multi_hand_landmarks:
            my_hand=self.results.multi_hand_landmarks[hand_no]

            for id,lm in enumerate(my_hand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmlist.append([id,cx,cy])
                if draw :
                    cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)


        return lmlist











def main():
    Ptime=0
    Ctime=0
    cap=cv2.VideoCapture(0)
    detector=handdetector()
    while True:
        ret,img=cap.read()
        img=detector.find_hands(img)
        lmlist=detector.find_positions(img)
        Ctime=time.time()
        fps=1/(Ctime-Ptime)
        Ptime=Ctime
        if len(lmlist) != 0:
            print(lmlist)

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,255,0),3)
        if cv2.waitKey(1)==13 :
            cv2.destroyAllWindows()
            break

if __name__=='__main__':
    main()                
