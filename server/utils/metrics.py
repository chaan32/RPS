# Metrics.py가 할일 
# JsonLines Loggger 클래스 정의
import json
import threading
from pathlib import Path

class JsonLinesLogger:
    def __init__ (self, path:str | Path): # 생성자
        # path 인자를 Path 객체로 통일 함 
        self.path = Path(path)

        # 부모 폴더 자동 생성
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # parents True : /a/b/c 했을 때 중간의 모든 폴더 다 생성ㅎ기
        # exist_ok True : 이미 존재해도 ㅇㅋ 
        

        # 동시에 로그 호출 시에 파일이 깨지지 않게 Lock을 걸어버림
        self._lock = threading.Lock()
    
    def log(self, data: dict) -> None :
        '''dict 한 개를 JSON 한 줄로 파일 끝에 append 해줌'''

        # dict -> json 문자열로 변환 함 
        line = json.dumps(data) + '\n'

        # lock 걸어서 동시 쓰기 방지 
        with self._lock : # 생성할 때 만들었던 락 사용
            with open(self.path, 'a') as f: # 파일 열기 append로 파일 열기
                f.write(line)
