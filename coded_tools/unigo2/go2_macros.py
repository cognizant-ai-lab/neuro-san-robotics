import os
import time
import platform

USE_REAL_ROBOT = False         # set True later on Ubuntu with robot
IFNAME = "en0"                 # replace with your NIC when on robot LAN

try:
    if USE_REAL_ROBOT:
        from unitree_sdk2_python.high_level import sport_client
except Exception as e:
    sport_client = None

class Go2Macros:
    def __init__(self, use_robot=False, ifname=None):
        self.use_robot = use_robot
        self.ifname = ifname
        if self.use_robot:
            # connect to the robot’s high-level (sports) client over CycloneDDS
            self.cli = sport_client.SportClient(ifname)
        else:
            self.cli = None

    def _log(self, msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def stand_up(self):
        if self.cli:
            self.cli.StandUp()
        self._log("Stand up")

    def lie_down(self):
        if self.cli:
            self.cli.LieDown()
        self._log("Lie down")

    def look_left(self):
        # Often implemented as a short attitude command (yaw) or a preset motion
        if self.cli:
            # example: small yaw left while standing
            self.cli.BalanceAttitude(roll=0.0, pitch=0.0, yaw=0.3, time=1.0)
        self._log("Look left (yaw +)")

    def look_right(self):
        if self.cli:
            self.cli.BalanceAttitude(roll=0.0, pitch=0.0, yaw=-0.3, time=1.0)
        self._log("Look right (yaw -)")

    def step_forward(self, vx=0.1, t=1.0):
        if self.cli:
            self.cli.VelocityMove(vx=vx, vy=0.0, yaw_rate=0.0, time=t)
        self._log(f"Step forward vx={vx} for {t}s")

    def shake(self):
        # “SpecialMotions” covers preset gestures like shake/rollover on supported firmware
        if self.cli:
            self.cli.SpecialMotions("shake")
        self._log("Shake")

    def roll_over(self):
        if self.cli:
            self.cli.SpecialMotions("rollover")
        self._log("Roll over")

if __name__ == "__main__":
    bot = Go2Macros(use_robot=USE_REAL_ROBOT, ifname=IFNAME if USE_REAL_ROBOT else None)
    bot.stand_up()
    time.sleep(1.0)
    bot.look_left(); time.sleep(0.5)
    bot.look_right(); time.sleep(0.5)
    bot.step_forward(0.2, 1.0); time.sleep(0.5)
    bot.shake(); time.sleep(1.0)
    bot.roll_over(); time.sleep(1.0)
    bot.lie_down()
