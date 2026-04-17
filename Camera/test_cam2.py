# Zero-Latency Version
import cv2 # 영상 처리를 위한 도구
import os # 컴퓨터 시스템 설정을 건드리기 위한 도구
import threading # 별도의 '전담 직원(스레드)'을 고용하기 위한 도구
from datetime import datetime # 시간을 가져오기 위한 도구

# 1. OpenCV 하부 엔진(FFMPEG)에게 "버퍼링 절대 하지 마!"라고 강제로 명령하는 설정입니다.
# 이 설정은 비디오 캡처를 시작하기 전에 미리 선언해야 효과가 있습니다.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|analyzeduration;0|probesize;32"

# 팀장님의 카메라 주소 (가벼운 stream2 사용)
rtsp_url = "rtsp://kkkhhhsssbrian:qpalzmxncbv@192.168.0.8:554/stream2"

# 2. 영상만 미친 듯이 퍼오는 '전담 직원' 클래스를 만듭니다.
class VideoStream:
    def __init__(self, src):
        # 파이프를 꽂습니다.
        self.stream = cv2.VideoCapture(src)
        # 바구니 크기를 1로 제한합니다.
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # 첫 번째 사진을 한 장 미리 퍼둡니다.
        self.ret, self.frame = self.stream.read()
        # 직원이 일을 멈출지 결정하는 신호등입니다.
        self.stopped = False

    def start(self):
        # "이제부터 이 직원(update 함수)은 별도의 공간에서 따로 일해!"라고 명령합니다.
        # daemon=True는 메인 프로그램이 꺼지면 이 직원도 같이 퇴근하라는 뜻입니다.
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        # 이 직원은 무한 루프를 돌며 오직 '최신 사진 퍼오기'만 반복합니다.
        while not self.stopped:
            # 바구니에 새 사진이 오자마자 바로 낚아채서 self.frame에 업데이트합니다.
            self.ret, self.frame = self.stream.read()

    def read(self):
        # 메인 프로그램이 "지금 가장 싱싱한 사진 줘!"라고 하면 즉시 손에 든 사진을 넘겨줍니다.
        return self.ret, self.frame

    def stop(self):
        # 직원에게 일을 그만하라고 신호를 보냅니다.
        self.stopped = True
        # 파이프를 뽑습니다.
        self.stream.release()

# 3. 메인 실행부 (CEO 관제 시스템 시작)
print("초저지연 모드로 카메라에 접속합니다...")

# 전담 직원을 고용하고 일을 시작(start)시킵니다.
v_stream = VideoStream(rtsp_url).start()

# 카메라가 제대로 열렸는지 확인합니다.
if not v_stream.stream.isOpened():
    print("❌ 연결 실패")
    exit()

print("✅ 멀티스레딩 최적화 완료!")

while True:
    # 전담 직원이 미리 퍼놓은 '가장 최신 사진'을 기다림 없이 바로 가져옵니다. (지연 제거의 핵심!)
    ret, frame = v_stream.read()
    
    # 가끔 전송 오류로 사진이 비어있으면 건너뜁니다.
    if not ret or frame is None:
        continue
        
    # 랩탑의 정확한 시간을 가져와서 영상에 노란색으로 도장을 찍습니다.
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
    # 화면에 출력합니다.
    cv2.imshow('Zero-Latency Mode', frame)
    
    # 'q'를 누르면 안전하게 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 작업이 끝나면 직원을 퇴근시키고 창을 닫습니다.
v_stream.stop()
cv2.destroyAllWindows()