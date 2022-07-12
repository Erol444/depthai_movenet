import depthai as dai

class CamControl:

    EXP_STEP = 500  # us
    ISO_STEP = 50
    LENS_STEP = 3

    lensPos = 150
    lensMin = 0
    lensMax = 255

    expTime = 2000
    expMin = 1
    expMax = 33000

    sensIso = 300
    sensMin = 100
    sensMax = 1600

    def __init__(self, controlQueue) -> None:
        self.q = controlQueue

    def clamp(self, num, v0, v1):
        return max(v0, min(num, v1))

    def check_key(self, key):
        if key in [ord(','), ord('.')]:
            if key == ord(','): self.lensPos -= self.LENS_STEP
            if key == ord('.'): self.lensPos += self.LENS_STEP
            self.lensPos = self.clamp(self.lensPos, self.lensMin, self.lensMax)
            print("Setting manual focus, lens position: ", self.lensPos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(self.lensPos)
            self.q.send(ctrl)
        elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
            if key == ord('i'): self.expTime -= self.EXP_STEP
            if key == ord('o'): self.expTime += self.EXP_STEP
            if key == ord('k'): self.sensIso -= self.ISO_STEP
            if key == ord('l'): self.sensIso += self.ISO_STEP
            self.expTime = self.clamp(self.expTime, self.expMin, self.expMax)
            self.sensIso = self.clamp(self.sensIso, self.sensMin, self.sensMax)
            print("Setting manual exposure, time: ", self.expTime, "iso: ", self.sensIso)
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(self.expTime, self.sensIso)
            self.q.send(ctrl)