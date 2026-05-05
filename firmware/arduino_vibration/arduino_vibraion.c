#include <WiFiS3.h>
#include <ArduinoMqttClient.h>

char ssid[] = "c3c3c3";
char pass[] = "castle1122eun";

// 브로커 IP와 포트는 분리해서 넣어야 함
const char broker[] = "192.168.0.14";   // <- 본사 서버 또는 MQTT 브로커 PC IP로 변경
int port = 1883;                        // <- 실제 포트로 변경

WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);

const int BUZZER_LEFT  = 8;
const int BUZZER_BACK  = 9;
const int BUZZER_RIGHT = 10;

void beepActive(int pin, int ms = 3000) {
    digitalWrite(pin, HIGH);
    delay(ms);
    digitalWrite(pin, LOW);
}

void beepAll(int ms = 3000) {
    digitalWrite(BUZZER_LEFT, HIGH);
    digitalWrite(BUZZER_BACK, HIGH);
    digitalWrite(BUZZER_RIGHT, HIGH);
    delay(ms);
    digitalWrite(BUZZER_LEFT, LOW);
    digitalWrite(BUZZER_BACK, LOW);
    digitalWrite(BUZZER_RIGHT, LOW);
}

void connectWiFi() {
    int status = WL_IDLE_STATUS;

    Serial.print("Connecting to WiFi");
    while (status != WL_CONNECTED) {
        status = WiFi.begin(ssid, pass);
        Serial.print(".");
        delay(3000);
    }

    delay(3000);

    Serial.println();
    Serial.println("WiFi connected");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
}

void subscribeTopics() {
    mqttClient.subscribe("crane/2/vibration");
    Serial.println("Subscribed to topic: crane/2/vibration");
}

void connectMQTT() {
    Serial.print("Connecting to MQTT broker... ");

    while (!mqttClient.connect(broker, port)) {
        Serial.print("failed, error code = ");
        Serial.println(mqttClient.connectError());
        delay(3000);
        Serial.print("Retrying MQTT... ");
    }

    Serial.println("connected");
    subscribeTopics();
}

void onMqttMessage(int messageSize) {
    String topic = mqttClient.messageTopic();
    String payload = "";

    while (mqttClient.available()) {
        payload += (char)mqttClient.read();
    }

    payload.trim();  // 혹시 공백/개행 들어오면 제거

    Serial.print("Topic: ");
    Serial.println(topic);
    Serial.print("Payload: ");
    Serial.println(payload);

    if (topic == "crane/2/vibration") {
        if (payload == "left") {
        Serial.println("LEFT buzzer ON");
        beepActive(BUZZER_LEFT);
        } 
        else if (payload == "back") {
        Serial.println("BACK buzzer ON");
        beepActive(BUZZER_BACK);
        } 
        else if (payload == "right") {
        Serial.println("RIGHT buzzer ON");
        beepActive(BUZZER_RIGHT);
        } 
        else if (payload == "all") {
        Serial.println("ALL buzzers ON");
        beepAll();
        } 
        else {
        Serial.println("Unknown payload");
        }
    }
}

void setup() {
    pinMode(BUZZER_LEFT, OUTPUT);
    pinMode(BUZZER_BACK, OUTPUT);
    pinMode(BUZZER_RIGHT, OUTPUT);

    digitalWrite(BUZZER_LEFT, LOW);
    digitalWrite(BUZZER_BACK, LOW);
    digitalWrite(BUZZER_RIGHT, LOW);

    Serial.begin(9600);
    // while (!Serial);  // 이 줄은 빼는 게 편함

    connectWiFi();

    mqttClient.onMessage(onMqttMessage);
    connectMQTT();
}

void loop() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi lost. Reconnecting...");
        connectWiFi();
        connectMQTT();
    }

    if (!mqttClient.connected()) {
        Serial.println("MQTT lost. Reconnecting...");
        connectMQTT();
    }

    mqttClient.poll();
}