// זבוב — קוד בסיס לרובוט (Arduino / ESP32)
// מקבל פקודות JSON מהאתר דרך USB (Web Serial, 115200) ומנפנף בשתי כנפיים עם סרוו.
//
// חיווט:  כנף שמאל -> פין 9 , כנף ימין -> פין 10 , מדידת סוללה -> A0 (דרך מחלק מתח)
// ספריות: ArduinoJson (v7).  ב-ESP32 להחליף את <Servo.h> ב-<ESP32Servo.h>.
//
// פקודות שהאתר שולח (שורה אחת לכל הודעה):
//   {"t":"cmd","arm":1,"fly":1,"yaw":0.1,"pitch":0.5,"roll":0,"climb":0,"hz":25,"ampL":0.8,"ampR":0.7,"auto":0}
//   {"t":"arm","on":1}  {"t":"takeoff"}  {"t":"land"}  {"t":"hover"}  {"t":"estop"}  {"t":"escape","heading":90}
// טלמטריה שהרובוט שולח בחזרה:
//   {"t":"tel","bat":87,"alt":0,"hdg":0}   ואופציונלית "sensors":{"eyeL":0,"eyeR":0,"antL":0,"antR":0,"loom":0}
//   {"t":"log","msg":"..."}

#include <Servo.h>
#include <ArduinoJson.h>

const int PIN_WING_L = 9;
const int PIN_WING_R = 10;
const int PIN_BAT = A0;
const float SERVO_MAX_HZ = 6.0;       // סרוו רגיל לא מסוגל לנפנף מהר יותר
const float SWING_DEG = 60.0;         // משרעת מקסימלית לכל צד
const unsigned long CMD_TIMEOUT = 500; // אם אין פקודה חצי שנייה — עוצרים

Servo wingL, wingR;
bool armed = false, flying = false;
float hz = 3, ampL = 0, ampR = 0;
float phase = 0;
unsigned long lastCmd = 0, lastTel = 0, lastLoop = 0;
String line;

void sendLog(const char* msg) {
  JsonDocument d; d["t"] = "log"; d["msg"] = msg;
  serializeJson(d, Serial); Serial.println();
}

void stopAll() {
  armed = false; flying = false; ampL = ampR = 0;
}

void handle(const String& s) {
  JsonDocument d;
  if (deserializeJson(d, s)) return;
  const char* t = d["t"] | "";
  if (!strcmp(t, "cmd")) {
    lastCmd = millis();
    armed = d["arm"] | 0;
    flying = d["fly"] | 0;
    hz = min((float)(d["hz"] | 3.0), SERVO_MAX_HZ);
    ampL = d["ampL"] | 0.0;
    ampR = d["ampR"] | 0.0;
  } else if (!strcmp(t, "estop")) {
    stopAll(); sendLog("עצירת חירום");
  } else if (!strcmp(t, "arm")) {
    armed = d["on"] | 0; sendLog(armed ? "חמוש" : "כבוי");
  } else if (!strcmp(t, "escape")) {
    sendLog("בריחה!");
  }
}

void setup() {
  Serial.begin(115200);
  wingL.attach(PIN_WING_L);
  wingR.attach(PIN_WING_R);
  wingL.write(90); wingR.write(90);
  sendLog("זבוב מוכן");
}

void loop() {
  // קריאת פקודות
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handle(line); line = ""; }
    else if (line.length() < 300) line += c;
  }

  // בטיחות: איבוד קשר = עצירה
  if (armed && millis() - lastCmd > CMD_TIMEOUT) { stopAll(); sendLog("אבד קשר — עצירה"); }

  // נפנוף כנפיים
  unsigned long now = millis();
  float dt = (now - lastLoop) / 1000.0; lastLoop = now;
  if (armed && (ampL > 0 || ampR > 0)) {
    phase += 2 * PI * hz * dt;
    if (phase > 2 * PI) phase -= 2 * PI;
    float s = sin(phase);
    wingL.write(90 + s * SWING_DEG * ampL);
    wingR.write(90 - s * SWING_DEG * ampR); // מראה: הכנף הימנית זזה הפוך
  } else {
    wingL.write(90); wingR.write(90);
  }

  // טלמטריה 5 פעמים בשנייה
  if (now - lastTel > 200) {
    lastTel = now;
    // התאם לפי מחלק המתח והסוללה שלך (כאן: 0..1023 -> 0..100%)
    int bat = map(analogRead(PIN_BAT), 600, 860, 0, 100);
    JsonDocument d;
    d["t"] = "tel";
    d["bat"] = constrain(bat, 0, 100);
    serializeJson(d, Serial); Serial.println();
  }
}
