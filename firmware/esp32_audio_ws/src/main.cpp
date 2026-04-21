// ESP32-S3-WROOM-1 + INMP441 I2S 마이크 → FastAPI WebSocket 서버 스트리머
//
// 프로토콜 (서버: input/audio/esp32_ws.py)
//   URL               ws://<맥북 IP>:1122/ws/audio
//   클라이언트 → 서버  16kHz / 16-bit signed LE / mono PCM 바이너리 프레임
//   서버 → 클라이언트  ready / silence / detection / error 형태의 JSON
//
// INMP441 ↔ ESP32-S3 배선 (아래 PIN_ 상수를 보드에 맞게 수정)
//   VDD  → 3V3
//   GND  → GND
//   L/R  → GND  (왼쪽 채널 사용)
//   WS   → PIN_I2S_WS
//   SCK  → PIN_I2S_SCK
//   SD   → PIN_I2S_SD
//
// 핵심 동작
//   1) Wi-Fi STA 모드 접속 (맥북과 같은 공유기)
//   2) WebSocket으로 /ws/audio 접속
//   3) I2S DMA로 오디오 읽기 → int32(24-bit MSB) → int16 축소 → sendBinary
//   4) 서버 JSON 결과는 콘솔에 프린트
//   5) 연결이 끊기면 자동 재접속 (지수 백오프)

#include <Arduino.h>
#include <WiFi.h>
#include <ArduinoWebsockets.h>
#include <driver/i2s.h>

// ── 사용자 설정 ────────────────────────────────────────────────────────
static const char* WIFI_SSID     = "c3c3c3";
static const char* WIFI_PASSWORD = "castle1122eun";

// 맥북 LAN IP (지금 값: 192.168.0.44). 공유기에서 고정 IP 잡아두는 걸 권장.
static const char* WS_URL = "ws://192.168.0.4:1122/ws/audio";

// INMP441 핀 (ESP32-S3-DevKitC-1 기준 예시 — 사용 보드에 맞춰 변경)
static constexpr int PIN_I2S_WS  = 5;   // LRCL
static constexpr int PIN_I2S_SCK = 6;   // BCLK
static constexpr int PIN_I2S_SD  = 4;   // DOUT

// 오디오 파라미터 (서버 프로토콜과 반드시 일치)
static constexpr int      SAMPLE_RATE      = 16000;
static constexpr i2s_port_t I2S_PORT       = I2S_NUM_0;
static constexpr int      DMA_BUF_COUNT    = 8;
static constexpr int      DMA_BUF_LEN      = 512;

// 한 번에 서버로 보낼 샘플 수. 1024 샘플 ≈ 64 ms 지연 / 2048 B 바이너리 프레임.
static constexpr size_t   CHUNK_SAMPLES    = 1024;


// ── 내부 상태 ──────────────────────────────────────────────────────────
using namespace websockets;

static WebsocketsClient  ws;
static int32_t           rawBuf[CHUNK_SAMPLES];   // INMP441 24-bit → 32-bit 슬롯
static int16_t           pcmBuf[CHUNK_SAMPLES];   // 서버로 보낼 int16
static unsigned long     nextReconnectMs = 0;
static unsigned long     reconnectBackoffMs = 500;  // 500ms부터 시작, 최대 10s까지 배수


// ── I2S 초기화 ─────────────────────────────────────────────────────────
static void setupI2S() {
  i2s_config_t cfg = {};
  cfg.mode                  = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate           = SAMPLE_RATE;
  cfg.bits_per_sample        = I2S_BITS_PER_SAMPLE_32BIT;   // INMP441은 24-bit를 32-bit 슬롯으로 전송
  cfg.channel_format         = I2S_CHANNEL_FMT_ONLY_LEFT;   // L/R 핀을 GND로 묶었을 때
  cfg.communication_format   = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags       = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count          = DMA_BUF_COUNT;
  cfg.dma_buf_len            = DMA_BUF_LEN;
  cfg.use_apll               = false;
  cfg.tx_desc_auto_clear     = false;
  cfg.fixed_mclk             = 0;

  i2s_pin_config_t pins = {};
  pins.bck_io_num   = PIN_I2S_SCK;
  pins.ws_io_num    = PIN_I2S_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num  = PIN_I2S_SD;

  esp_err_t e;
  e = i2s_driver_install(I2S_PORT, &cfg, 0, nullptr);
  if (e != ESP_OK) { Serial.printf("i2s_driver_install 실패: %d\n", e); while (true) delay(1000); }
  e = i2s_set_pin(I2S_PORT, &pins);
  if (e != ESP_OK) { Serial.printf("i2s_set_pin 실패: %d\n", e); while (true) delay(1000); }
  i2s_zero_dma_buffer(I2S_PORT);
  Serial.println("[I2S] 초기화 완료");
}

// INMP441 1 샘플 = 32-bit 워드(상위 24-bit에 PCM, MSB 정렬)
// 상위 16-bit만 추출하면 int16 signed PCM이 된다.
static size_t readI2S16(int16_t* out, size_t samples) {
  size_t bytesRead = 0;
  i2s_read(I2S_PORT, rawBuf, samples * sizeof(int32_t), &bytesRead, portMAX_DELAY);
  const size_t n = bytesRead / sizeof(int32_t);
  for (size_t i = 0; i < n; ++i) {
    out[i] = (int16_t)(rawBuf[i] >> 16);
  }
  return n;
}


// ── WebSocket 콜백 ────────────────────────────────────────────────────
static void onWsMessage(WebsocketsMessage msg) {
  Serial.print("[서버] ");
  Serial.println(msg.data());
}

static void onWsEvent(WebsocketsEvent event, String data) {
  switch (event) {
    case WebsocketsEvent::ConnectionOpened:
      Serial.println("[WS] 접속됨");
      reconnectBackoffMs = 500;  // 성공 시 백오프 리셋
      break;
    case WebsocketsEvent::ConnectionClosed:
      Serial.println("[WS] 끊김");
      break;
    case WebsocketsEvent::GotPing:
      // 라이브러리가 자동으로 pong 응답
      break;
    default:
      break;
  }
}


// ── Wi-Fi ─────────────────────────────────────────────────────────────
static void connectWiFi() {
  Serial.printf("[WiFi] %s 접속 중", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);   // 오디오 스트리밍 지연 줄이기
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
    if (millis() - start > 20000) {    // 20초 넘어가면 재시도
      Serial.println(" 타임아웃, 재시도");
      WiFi.disconnect(true);
      delay(500);
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      start = millis();
    }
  }
  Serial.printf("\n[WiFi] 연결됨  IP=%s  RSSI=%d dBm\n",
                WiFi.localIP().toString().c_str(), WiFi.RSSI());
}


// ── WebSocket 접속/재접속 ─────────────────────────────────────────────
static bool connectWS() {
  ws.onMessage(onWsMessage);
  ws.onEvent(onWsEvent);
  Serial.printf("[WS] 접속 시도: %s\n", WS_URL);
  bool ok = ws.connect(WS_URL);
  if (!ok) Serial.println("[WS] 접속 실패");
  return ok;
}

static void maintainConnection() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] 끊김 → 재접속");
    connectWiFi();
  }
  if (ws.available()) return;

  if (millis() < nextReconnectMs) return;
  if (!connectWS()) {
    reconnectBackoffMs = min<unsigned long>(reconnectBackoffMs * 2, 10000);
    nextReconnectMs = millis() + reconnectBackoffMs;
    Serial.printf("[WS] %lu ms 후 재시도\n", reconnectBackoffMs);
  }
}


// ── Arduino entry points ──────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\n=== ESP32-S3 Audio WebSocket Client ===");
  connectWiFi();
  setupI2S();
  connectWS();
}

void loop() {
  maintainConnection();

  if (!ws.available()) {
    // 접속 전에도 마이크 데이터를 계속 읽어 DMA 오버플로우를 피한다
    size_t discarded = 0;
    i2s_read(I2S_PORT, rawBuf, sizeof(rawBuf), &discarded, 0);
    delay(10);
    return;
  }

  // 1) 마이크에서 한 청크 읽기 (blocking, 약 64 ms)
  size_t n = readI2S16(pcmBuf, CHUNK_SAMPLES);
  if (n == 0) { ws.poll(); return; }

  // 2) int16 PCM 바이너리 프레임으로 전송
  const size_t bytes = n * sizeof(int16_t);
  bool sent = ws.sendBinary(reinterpret_cast<const char*>(pcmBuf), bytes);
  if (!sent) {
    Serial.println("[WS] sendBinary 실패");
    ws.close();
    return;
  }

  // 3) 서버가 보낸 메시지/핑 처리
  ws.poll();
}
