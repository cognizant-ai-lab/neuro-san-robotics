import os
import time
import platform
import traceback

USE_REAL_ROBOT = True
IFNAME = "eth0"                     # "eth0" if you're on wired

# --- Recommend setting these in your shell profile too ---
os.environ.setdefault("CYCLONEDDS_NETWORK_INTERFACE", IFNAME)
# If you installed CycloneDDS to a custom prefix, export CYCLONEDDS_HOME in your shell.
# os.environ.setdefault("CYCLONEDDS_HOME", "/home/unitree/exp/neuro-san-robotics/cyclonedds/install")

try:
    if USE_REAL_ROBOT:
        # Unitree SDK imports
        from unitree_sdk2_python.unitree_sdk2py.go2.sport import sport_client
        from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelFactoryInitialize
except Exception:
    sport_client = None
    ChannelFactoryInitialize = None


def _coerce_status(ret):
    """
    Unitree SDK sometimes returns just 'code' and sometimes '(code, data)'.
    This helper normalizes that to (code, data_or_None).
    """
    if isinstance(ret, tuple) and len(ret) >= 1:
        return ret[0], (None if len(ret) == 1 else ret[1])
    return ret, None


class Go2Macros:
    def __init__(self, use_robot=USE_REAL_ROBOT, ifname=IFNAME):
        self.use_robot = use_robot
        self.ifname = ifname
        self.cli = None

        if not self.use_robot or sport_client is None:
            self._log("⚙️ Running in simulation/offline mode (no robot)")
            return

        try:
            # --- Initialize DDS channel (matches go2_sport_client.py) ---
            self._log("🔍 Initializing ChannelFactory")
            ChannelFactoryInitialize(0)

            # --- Initialize Sport client (matches go2_sport_client.py) ---
            self._log("🔍 Initializing SportClient")
            self.cli = sport_client.SportClient()
            self.cli.SetTimeout(10.0)
            self.cli.Init()
            self._log("✅ SportClient initialized and ready")

        except Exception as e:
            self._log(f"❌ Failed to initialize: {e}")
            traceback.print_exc()

    def _log(self, msg: str):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # ----------------------------
    # BASIC MOTIONS
    # ----------------------------
    def damp(self):
        if self.cli:
            self.cli.Damp()
        self._log("Damp")

    def balance_stand(self):
        if self.cli:
            self.cli.BalanceStand()
        self._log("Balance stand")

    def stop_move(self):
        if self.cli:
            self.cli.StopMove()
        self._log("Stop move")

    def stand_up(self):
        if self.cli:
            self.cli.StandUp()
        self._log("Stand up")

    def lie_down(self):
        if self.cli:
            self.cli.StandDown()
        self._log("Lie down")

    def recovery_stand(self):
        if self.cli:
            self.cli.RecoveryStand()
        self._log("Recovery stand")

    def sit(self):
        if self.cli:
            self.cli.Sit()
        self._log("Sit")

    def rise_sit(self):
        if self.cli:
            self.cli.RiseSit()
        self._log("Rise sit")

    def sit_rise(self):
        if self.cli:
            self.cli.Sit()
            self.cli.RiseSit()
        self._log("Sit rise")

    # ----------------------------
    # ORIENTATION / LOOK DIRECTION
    # ----------------------------
    def euler(self, roll=0.0, pitch=0.0, yaw=0.0):
        """Set body orientation with roll, pitch, yaw."""
        if self.cli:
            self.cli.Euler(roll=roll, pitch=pitch, yaw=yaw)
        self._log(f"Euler orientation (roll={roll}, pitch={pitch}, yaw={yaw})")

    def look_left(self):
        if self.cli:
            self.cli.Euler(roll=0.0, pitch=0.0, yaw=0.3)
            time.sleep(1.0)
            self.cli.Euler(0.0, 0.0, 0.0)
        self._log("Look left (yaw +)")

    def look_right(self):
        if self.cli:
            self.cli.Euler(roll=0.0, pitch=0.0, yaw=-0.3)
            time.sleep(1.0)
            self.cli.Euler(0.0, 0.0, 0.0)
        self._log("Look right (yaw -)")

    # ----------------------------
    # LOCOMOTION
    # ----------------------------
    def move(self, vx=0.0, vy=0.0, vyaw=0.0):
        """Continuous movement command. Call stop_move() to stop."""
        if self.cli:
            self.cli.Move(vx=vx, vy=vy, vyaw=vyaw)
        self._log(f"Move (vx={vx}, vy={vy}, vyaw={vyaw})")

    def step_forward(self, vx=0.1, t=1.0):
        if self.cli:
            self.cli.Move(vx=vx, vy=0.0, vyaw=0.0)
            time.sleep(t)
            self.cli.StopMove()
        self._log(f"Step forward vx={vx} for {t}s")

    def step_backward(self, vx=-0.1, t=1.0):
        if self.cli:
            self.cli.Move(vx=vx, vy=0.0, vyaw=0.0)
            time.sleep(t)
            self.cli.StopMove()
        self._log(f"Step backward vx={vx} for {t}s")

    def speed_level(self, level):
        """Set speed level."""
        if self.cli:
            self.cli.SpeedLevel(level)
        self._log(f"Speed level set to {level}")

    # ----------------------------
    # SPECIAL MOTIONS / EXPRESSIONS
    # ----------------------------
    def shake(self):
        if self.cli:
            self.cli.Hello()
        self._log("Shake / Hello motion")

    def stretch(self):
        if self.cli:
            self.cli.Stretch()
        self._log("Stretch motion")

    def content(self):
        if self.cli:
            self.cli.Content()
        self._log("Content motion")

    def dance(self):
        if self.cli:
            self.cli.Dance1()
        self._log("Dance 1 motion")

    def dance1(self):
        if self.cli:
            self.cli.Dance1()
        self._log("Dance 1 motion")

    def dance2(self):
        if self.cli:
            self.cli.Dance2()
        self._log("Dance 2 motion")

    def pose(self, flag):
        if self.cli:
            self.cli.Pose(flag)
        self._log(f"Pose motion (flag={flag})")

    def scrape(self):
        if self.cli:
            self.cli.Scrape()
        self._log("Scrape motion")

    def heart_pose(self):
        if self.cli:
            self.cli.Heart()
        self._log("Heart pose motion")

    # ----------------------------
    # FLIPS AND ACROBATICS
    # ----------------------------
    def front_flip(self):
        if self.cli:
            self.cli.FrontFlip()
        self._log("Front flip motion")

    def front_jump(self):
        if self.cli:
            self.cli.FrontJump()
        self._log("Front jump motion")

    def front_pounce(self):
        if self.cli:
            self.cli.FrontPounce()
        self._log("Front pounce motion")

    def left_flip(self):
        if self.cli:
            self.cli.LeftFlip()
        self._log("Left flip motion")

    def backflip(self):
        if self.cli:
            self.cli.BackFlip()
        self._log("Backflip motion")

    def hand_stand(self, flag):
        if self.cli:
            self.cli.HandStand(flag)
        self._log(f"Hand stand (flag={flag})")

    # ----------------------------
    # GAIT AND WALKING MODES
    # ----------------------------
    def static_walk(self):
        if self.cli:
            self.cli.StaticWalk()
        self._log("Static walk gait")

    def trot_run(self):
        if self.cli:
            self.cli.TrotRun()
        self._log("Trot run gait")

    def free_walk(self):
        if self.cli:
            self.cli.FreeWalk()
        self._log("Free walk")

    def free_bound(self, flag):
        if self.cli:
            self.cli.FreeBound(flag)
        self._log(f"Free bound (flag={flag})")

    def free_jump(self, flag):
        if self.cli:
            self.cli.FreeJump(flag)
        self._log(f"Free jump (flag={flag})")

    def free_avoid(self, flag):
        if self.cli:
            self.cli.FreeAvoid(flag)
        self._log(f"Free avoid (flag={flag})")

    def classic_walk(self, flag):
        if self.cli:
            self.cli.ClassicWalk(flag)
        self._log(f"Classic walk (flag={flag})")

    def walk_upright(self, flag):
        if self.cli:
            self.cli.WalkUpright(flag)
        self._log(f"Walk upright (flag={flag})")

    def cross_step(self, flag):
        if self.cli:
            self.cli.CrossStep(flag)
        self._log(f"Cross step (flag={flag})")

    # ----------------------------
    # CONFIGURATION AND SETTINGS
    # ----------------------------
    def switch_joystick(self, on):
        if self.cli:
            self.cli.SwitchJoystick(on)
        self._log(f"Switch joystick (on={on})")

    def auto_recovery_set(self, enabled):
        if self.cli:
            self.cli.AutoRecoverySet(enabled)
        self._log(f"Auto recovery set (enabled={enabled})")

    def auto_recovery_get(self):
        if self.cli:
            code, data = self.cli.AutoRecoveryGet()
            self._log(f"Auto recovery get: code={code}, data={data}")
            return code, data
        return None, None

    def switch_avoid_mode(self):
        if self.cli:
            self.cli.SwitchAvoidMode()
        self._log("Switch avoid mode")

    # ----------------------------
    # WRAPPER FOR SAFE SHUTDOWN
    # ----------------------------
    def shutdown(self):
        if self.cli:
            try:
                self.cli.StopMove()
            except Exception:
                pass
            try:
                self.cli.StandDown()
            except Exception:
                pass
        self._log("Shutdown complete")


# if __name__ == "__main__":
#    bot = Go2Macros(use_robot=USE_REAL_ROBOT, ifname=IFNAME if USE_REAL_ROBOT else None)

#    bot.stand_up()
#    time.sleep(1.0)

#    bot.look_left(); time.sleep(0.5)
#    bot.look_right(); time.sleep(0.5)

#    bot.step_forward(0.2, 1.0)
#    time.sleep(0.5)

#    bot.shake(); time.sleep(1.0)
#    bot.dance(); time.sleep(1.0)
#    bot.backflip(); time.sleep(1.0)
#    bot.heart_pose(); time.sleep(1.0)

#    bot.lie_down()
#    bot.shutdown()
