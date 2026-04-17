import cv2  # 비전 AI의 핵심이자, 영상을 다루는 가장 강력한 도구(라이브러리)를 불러옵니다.
from datetime import datetime  # 랩탑(서버)의 정확한 현재 시간을 가져오기 위한 도구를 불러옵니다.

# 1. 카메라 수도관 주소 설정
# 팀장님이 세팅하신 아이디, 비번, IP 주소를 넣는 곳입니다. 
# 끝부분의 stream2는 실시간 분석에 유리한 가벼운 영상(저지연)을 의미합니다.
rtsp_url = "rtsp://kkkhhhsssbrian:qpalzmxncbv@192.168.0.8:554/stream2" 

print("카메라와 연결을 시도합니다...")

# 2. 수도관 연결 (파이프 꽂기)
# cv2.VideoCapture는 지정한 주소(rtsp_url)로 가서 영상을 빨아들이는 파이프를 생성합니다.
cap = cv2.VideoCapture(rtsp_url)

# 3. 실시간 최적화 (버퍼링 금지)
# CAP_PROP_BUFFERSIZE를 1로 설정하여, 과거 영상을 메모리에 쌓아두지 않고 
# 항상 방금 도착한 '최신 장면 1장'만 가져오게 강제합니다. (1초 지연 현상 해결의 핵심!)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 4. 연결 상태 확인
# 파이프가 제대로 꽂혔는지(isOpened) 확인합니다. 
# 안 꽂혔다면 에러 메시지를 띄우고 프로그램을 강제 종료(exit)합니다.
if not cap.isOpened():
    print("❌ 연결 실패: 주소나 네트워크 상태를 다시 확인해주세요.")
    exit()

print("✅ 통신 성공! (자체 타임스탬프 모드 작동 중)")

# 5. 무한 반복문 (계속해서 실시간 영상을 받아오는 과정)
while True:
    # cap.read()는 파이프에서 물(영상)을 한 바가지 퍼옵니다.
    # ret: 성공적으로 퍼왔는지 여부 (True/False)
    # frame: 퍼온 사진 1장 (수많은 픽셀 숫자로 이루어진 배열 데이터)
    ret, frame = cap.read()
    
    # 만약 영상을 퍼오는 데 실패했다면(카메라 전원이 꺼졌거나 와이파이가 끊겼다면) 반복문을 탈출합니다.
    if not ret:
        print("영상이 끊겼습니다.")
        break
        
    # ---------- [시간 도장 찍기 핵심 구역] ----------
    # 6. 현재 시간을 사람이 읽기 편한 텍스트로 만들기
    # datetime.now()로 시간을 가져와서 "%Y-%m-%d %H:%M:%S" 포맷(연-월-일 시:분:초)으로 변환합니다.
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 7. 영상(frame) 위에 시간 글씨 쓰기
    # cv2.putText(도화지, 쓸 글씨, (가로위치, 세로위치), 폰트종류, 글자크기, 색상, 선두께)
    # 여기서 (0, 255, 255)는 노란색을 의미합니다. (OpenCV는 색상을 RGB가 아니라 BGR 순서로 씁니다)
    cv2.putText(frame, current_time, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    # ------------------------------------------------
        
    # 8. 완성된 사진을 화면에 띄우기
    # 'Vision Test'라는 이름의 창에 시간 도장이 찍힌 최신 사진(frame)을 띄워줍니다.
    # 이 과정이 1초에 수십 번 반복되면서 우리 눈에는 '자연스러운 동영상'으로 보이게 됩니다.
    cv2.imshow('Vision Test', frame)
    
    # 9. 종료 조건 (비상 탈출구)
    # cv2.waitKey(1)은 1밀리초(0.001초) 동안 키보드 입력이 있는지 짧게 기다립니다.
    # '& 0xFF == ord('q')'는 사용자가 누른 키가 영문 'q'인지 확인하는 공식입니다. q를 누르면 종료됩니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 10. 뒷정리 (메모리 관리)
# 탈출했으면 꽂아뒀던 파이프를 뽑고(release), 띄워놨던 창을 모두 닫아서(destroyAllWindows) 
# 컴퓨터의 자원을 깔끔하게 반환합니다.
cap.release()
cv2.destroyAllWindows()