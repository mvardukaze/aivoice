// მიკროფონის PCM-ს უცვლელად აგროვებს (დანაკარგის გარეშე — MediaRecorder-ის opus-ისგან განსხვავებით)
// და მთავარ ნაკადს დონის ინდიკატორისთვის პიკს უგზავნის.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.recording = false;
    this.port.onmessage = (e) => { this.recording = e.data === "start"; };
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      let peak = 0;
      for (let i = 0; i < ch.length; i++) { const v = Math.abs(ch[i]); if (v > peak) peak = v; }
      this.port.postMessage({ peak, data: this.recording ? ch.slice(0) : null });
    }
    return true;
  }
}
registerProcessor("capture", Capture);
