# Backend / AI Backend Portfolio Site

이 폴더는 포트폴리오 제출/공유용 정적 웹페이지입니다.

## 열기

`index.html`을 브라우저로 열면 바로 확인할 수 있습니다.

```bash
open /Users/haechan/Desktop/pobiga/ai/ai_project/portfolio_site/index.html
```

또는 로컬 서버로 확인하려면:

```bash
cd /Users/haechan/Desktop/pobiga/ai/ai_project/portfolio_site
python -m http.server 8088
```

브라우저에서 `http://localhost:8088`로 접속하면 됩니다.

## 구성

- `index.html`: 포트폴리오 페이지 본문
- `styles.css`: 반응형 레이아웃과 시각 디자인
- `script.js`: 성능 개선 차트/표 렌더링
- `assets/`: 검증 영상, 이미지, CSV 근거 자료

## 핵심 메시지

- 개인 포트폴리오 홈 형태로 구성
- About, Tech Stack, Selected Projects, Featured Project, Metrics, Next Step 섹션 포함
- 대표 프로젝트는 Unity RTSP 기반 지게차 충돌 위험 예측 시스템
- 기술 스택은 Backend, Realtime Media, Computer Vision, Performance, Infra, Frontend로 분류
- 초기 3.145 FPS에서 최대 10.148 FPS까지 개선한 성능 지표 포함

## 다른 프로젝트 추가 방법

`index.html`의 `Selected Projects` 섹션에서 `project-card` 블록을 복사해 프로젝트 제목, 설명, 기술 태그만 바꾸면 됩니다.
