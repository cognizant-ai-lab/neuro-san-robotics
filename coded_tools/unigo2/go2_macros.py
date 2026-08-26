import os
import time
import platform
import traceback
import threading
import math

USE_REAL_ROBOT = True
IFNAME = (
    os.environ.get("GO2_NETWORK_INTERFACE")
    or os.environ.get("CYCLONEDDS_NETWORK_INTERFACE")
    or "eth0"
)

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

try:
    if USE_REAL_ROBOT:
        from unitree_sdk2_python.unitree_sdk2py.go2.obstacles_avoid import (
            obstacles_avoid_client,
        )
except Exception:
    try:
        from unitree_sdk2py.go2.obstacles_avoid import obstacles_avoid_client
    except Exception:
        obstacles_avoid_client = None


_ROBOT_INIT_STATE = {
    "attempted": False,
    "available": True,
    "error": None,
    "reported_disabled": False,
    "client": None,
    "avoidance_client": None,
    "channel_initialized": False,
    "last_failure_at": 0.0,
    "locomotion_ready": False,
}
_ROBOT_INIT_LOCK = threading.Lock()


def _coerce_status(ret):
    """
    Unitree SDK sometimes returns just 'code' and sometimes '(code, data)'.
    This helper normalizes that to (code, data_or_None).
    """
    if isinstance(ret, tuple) and len(ret) >= 1:
        return ret[0], (None if len(ret) == 1 else ret[1])
    return ret, None


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


class Go2Macros:
    def __init__(self, use_robot=USE_REAL_ROBOT, ifname=IFNAME):
        self.use_robot = use_robot
        self.ifname = ifname
        self.cli = None
        self.avoidance_cli = None
        self.available = False
        self.last_recovery_error = None
        self._move_log_interval_s = _env_float("GO2_MOVE_LOG_INTERVAL_SECONDS", 1.0)
        self._last_move_log_at = 0.0

        if not self.use_robot or sport_client is None or ChannelFactoryInitialize is None:
            self._log("⚙️ Running in simulation/offline mode (no robot)")
            return

        with _ROBOT_INIT_LOCK:
            cached_client = _ROBOT_INIT_STATE.get("client")
            if cached_client is not None and _ROBOT_INIT_STATE["available"]:
                self.cli = cached_client
                self.avoidance_cli = _ROBOT_INIT_STATE.get("avoidance_client")
                self.available = True
                return

            if _ROBOT_INIT_STATE["attempted"] and not _ROBOT_INIT_STATE["available"]:
                retry_after_s = max(0.0, _env_float("GO2_INIT_RETRY_SECONDS", 2.0))
                elapsed_s = time.monotonic() - float(_ROBOT_INIT_STATE.get("last_failure_at") or 0.0)
                if elapsed_s < retry_after_s:
                    if not _ROBOT_INIT_STATE["reported_disabled"]:
                        self._log(
                            "⚙️ Robot control temporarily unavailable after prior "
                            f"initialization failure: {_ROBOT_INIT_STATE['error']}"
                        )
                        _ROBOT_INIT_STATE["reported_disabled"] = True
                    return

                if _ROBOT_INIT_STATE["reported_disabled"]:
                    self._log(
                        "🔁 Retrying robot control initialization after prior failure"
                    )

            try:
                if not _ROBOT_INIT_STATE["channel_initialized"]:
                    # DDS is process-global. Initializing it more than once can fail when
                    # Flask, vision, deferred actions, and nav all create Go2 helpers.
                    self._log("🔍 Initializing ChannelFactory")
                    if self.ifname:
                        ChannelFactoryInitialize(0, self.ifname)
                    else:
                        ChannelFactoryInitialize(0)
                    _ROBOT_INIT_STATE["channel_initialized"] = True

                # --- Initialize Sport client (matches go2_sport_client.py) ---
                self._log("🔍 Initializing SportClient")
                self.cli = sport_client.SportClient()
                self.cli.SetTimeout(10.0)
                self.cli.Init()
                self._configure_startup_motion_modes()
                self.available = True
                _ROBOT_INIT_STATE.update(
                    {
                        "attempted": True,
                        "available": True,
                        "error": None,
                        "reported_disabled": False,
                        "client": self.cli,
                        "avoidance_client": self.avoidance_cli,
                        "channel_initialized": True,
                        "last_failure_at": 0.0,
                        "locomotion_ready": False,
                    }
                )
                self._log("✅ SportClient initialized and ready")

            except Exception as e:
                _ROBOT_INIT_STATE.update(
                    {
                        "attempted": True,
                        "available": False,
                        "error": str(e),
                        "reported_disabled": False,
                        "client": None,
                        "avoidance_client": None,
                        "last_failure_at": time.monotonic(),
                        "locomotion_ready": False,
                    }
                )
                self._log(f"❌ Failed to initialize: {e}")
                traceback.print_exc()

    def _configure_startup_motion_modes(self):
        """Disable firmware avoidance so app-level navigation owns locomotion."""
        if not self.cli:
            return

        free_avoid = getattr(self.cli, "FreeAvoid", None)
        if callable(free_avoid):
            self._call("FreeAvoid", free_avoid, False)
        else:
            self._log("Free avoid startup disable skipped: SDK method unavailable")

        if obstacles_avoid_client is not None:
            try:
                self.avoidance_cli = obstacles_avoid_client.ObstaclesAvoidClient()
                self.avoidance_cli.SetTimeout(3.0)
                self.avoidance_cli.Init()
                self._call(
                    "ObstaclesAvoid.SwitchSet",
                    self.avoidance_cli.SwitchSet,
                    False,
                )
                _code, enabled = _coerce_status(self.avoidance_cli.SwitchGet())
                if enabled in (False, 0, "0", "false", "False"):
                    self._log("Onboard obstacle avoidance disabled on init")
                else:
                    self._log(
                        "⚠️ Onboard obstacle avoidance disable could not be verified: "
                        f"{enabled!r}"
                    )
            except Exception as exc:
                self.avoidance_cli = None
                self._log(f"⚠️ Onboard obstacle avoidance disable failed: {exc}")
        else:
            self._log("Onboard obstacle avoidance service unavailable in SDK")

    def _log(self, msg: str):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _log_move_command(self, vx: float, vy: float, vyaw: float):
        if self._move_log_interval_s < 0.0:
            return

        now = time.monotonic()
        if (
            self._move_log_interval_s <= 0.0
            or now - self._last_move_log_at >= self._move_log_interval_s
        ):
            self._last_move_log_at = now
            self._log(f"Move (vx={vx:.3f}, vy={vy:.3f}, vyaw={vyaw:.3f})")

    def _call(self, label: str, fn, *args, **kwargs):
        ret = fn(*args, **kwargs)
        code, _data = _coerce_status(ret)
        if code not in (0, None):
            self._log(f"⚠️ {label} returned {ret!r}")
        return ret

    def _prepare_locomotion(self):
        if not self.cli:
            return False

        recovery_ret = self._call("RecoveryStand", self.cli.RecoveryStand)
        time.sleep(_env_float("GO2_RECOVERY_STAND_SETTLE_SECONDS", 1.5))
        balance_ret = self._call("BalanceStand", self.cli.BalanceStand)
        time.sleep(_env_float("GO2_BALANCE_STAND_SETTLE_SECONDS", 0.5))
        recovery_code, _ = _coerce_status(recovery_ret)
        balance_code, _ = _coerce_status(balance_ret)
        ready = recovery_code in (0, None) and balance_code in (0, None)
        _ROBOT_INIT_STATE["locomotion_ready"] = ready
        return ready

    def ensure_locomotion_ready(self):
        """Prepare the robot once before accepting continuous navigation commands."""
        if not self.cli:
            return False
        if _ROBOT_INIT_STATE.get("locomotion_ready"):
            return True
        with _ROBOT_INIT_LOCK:
            if _ROBOT_INIT_STATE.get("locomotion_ready"):
                return True
            self._log("Preparing locomotion for navigation")
            return self._prepare_locomotion()

    def recover_locomotion(self):
        """Re-enter a known locomotion mode after accepted commands stop moving."""
        if not self.cli:
            self.last_recovery_error = "SportClient is unavailable"
            return False
        with _ROBOT_INIT_LOCK:
            self._log("Recovering locomotion after unacknowledged motion commands")
            try:
                self._call("StopMove", self.cli.StopMove)
            finally:
                _ROBOT_INIT_STATE["locomotion_ready"] = False
            ready = self._prepare_locomotion()
            self.last_recovery_error = (
                None if ready else "RecoveryStand or BalanceStand was rejected"
            )
            return ready

    def reinitialize_locomotion(self):
        """Replace the SportClient and prepare locomotion on the existing DDS domain.

        CycloneDDS initialization is process-global and must not be repeated.  A
        fresh SportClient, however, replaces stale request/reply state without
        touching navigation pose, localization, or route state held by NavCore.
        """
        if sport_client is None or not _ROBOT_INIT_STATE.get("channel_initialized"):
            self.last_recovery_error = "DDS channel or SportClient module is unavailable"
            return False

        with _ROBOT_INIT_LOCK:
            self._log("Reinitializing SportClient after unverified locomotion recovery")
            previous_client = self.cli
            if previous_client is not None:
                try:
                    self._call("StopMove", previous_client.StopMove)
                except Exception as exc:
                    self._log(f"⚠️ StopMove before SportClient replacement failed: {exc}")

            self.available = False
            self.cli = None
            self.avoidance_cli = None
            _ROBOT_INIT_STATE.update(
                {
                    "available": False,
                    "client": None,
                    "avoidance_client": None,
                    "locomotion_ready": False,
                }
            )
            try:
                replacement = sport_client.SportClient()
                replacement.SetTimeout(10.0)
                replacement.Init()
                self.cli = replacement
                self._configure_startup_motion_modes()
                self.available = True
                _ROBOT_INIT_STATE.update(
                    {
                        "attempted": True,
                        "available": True,
                        "error": None,
                        "reported_disabled": False,
                        "client": replacement,
                        "avoidance_client": self.avoidance_cli,
                        "last_failure_at": 0.0,
                        "locomotion_ready": False,
                    }
                )
                ready = self._prepare_locomotion()
                if not ready:
                    raise RuntimeError("RecoveryStand or BalanceStand was rejected")
                self.last_recovery_error = None
                self._log("✅ SportClient reinitialized; awaiting measured-motion verification")
                return True
            except Exception as exc:
                self.available = False
                self.last_recovery_error = str(exc)
                _ROBOT_INIT_STATE.update(
                    {
                        "attempted": True,
                        "available": False,
                        "error": str(exc),
                        "reported_disabled": False,
                        "client": None,
                        "avoidance_client": None,
                        "last_failure_at": time.monotonic(),
                        "locomotion_ready": False,
                    }
                )
                self._log(f"❌ SportClient reinitialization failed: {exc}")
                return False

    def _timed_move(
        self,
        vx: float,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration_s: float = 1.0,
        period_s: float | None = None,
    ):
        period_s = period_s or _env_float("GO2_MOVE_COMMAND_PERIOD", 0.2)
        period_s = max(0.05, period_s)
        iterations = max(1, math.ceil(max(0.0, duration_s) / period_s))

        if self.cli:
            for _ in range(iterations):
                self._call("Move", self.cli.Move, vx=vx, vy=vy, vyaw=vyaw)
                time.sleep(period_s)
            self._call("StopMove", self.cli.StopMove)

        self._log(f"Timed move (vx={vx}, vy={vy}, vyaw={vyaw}) for {duration_s}s")

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
    TURN_YAW_RATE_RPS = 0.6

    def turn(self, angle_deg: float = 90.0):
        """
        Rotate the whole robot in place.

        Distinct from look_left/look_right, which tilt the body on its Euler
        axes and leave the robot facing the same way. Positive turns left.
        """
        rate = abs(self.TURN_YAW_RATE_RPS)
        duration = abs(math.radians(angle_deg)) / rate
        self._timed_move(
            vx=0.0,
            vy=0.0,
            vyaw=rate if angle_deg >= 0.0 else -rate,
            duration_s=duration,
        )
        self._log(f"Turned {angle_deg:.0f} degrees")

    def turn_left(self, angle_deg: float = 90.0):
        """Rotate in place to the left."""
        self.turn(abs(angle_deg))

    def turn_right(self, angle_deg: float = 90.0):
        """Rotate in place to the right."""
        self.turn(-abs(angle_deg))

    def move(self, vx=0.0, vy=0.0, vyaw=0.0):
        """Continuous movement command. Call stop_move() to stop."""
        command_active = abs(vx) > 1e-3 or abs(vy) > 1e-3 or abs(vyaw) > 1e-3
        if command_active and not self.ensure_locomotion_ready():
            raise RuntimeError("Robot locomotion could not be prepared")
        if self.cli:
            ret = self._call("Move", self.cli.Move, vx=vx, vy=vy, vyaw=vyaw)
            code, _ = _coerce_status(ret)
            if code not in (0, None):
                raise RuntimeError(f"Robot Move command failed with status {ret!r}")
        self._log_move_command(vx, vy, vyaw)

    def step_forward(self, vx=None, t=None):
        vx = _env_float("GO2_STEP_FORWARD_SPEED", 0.45) if vx is None else vx
        t = (
            _env_float(
                "GO2_STEP_FORWARD_DURATION_SECONDS",
                _env_float("GO2_STEP_DURATION_SECONDS", 0.8),
            )
            if t is None
            else t
        )
        self._prepare_locomotion()
        self._timed_move(vx=vx, vy=0.0, vyaw=0.0, duration_s=t)
        self._log(f"Step forward vx={vx} for {t}s")

    def step_backward(self, vx=None, t=None):
        vx = _env_float("GO2_STEP_BACKWARD_SPEED", -0.25) if vx is None else vx
        t = (
            _env_float(
                "GO2_STEP_BACKWARD_DURATION_SECONDS",
                _env_float("GO2_STEP_DURATION_SECONDS", 2.0),
            )
            if t is None
            else t
        )
        self._prepare_locomotion()
        self._timed_move(vx=vx, vy=0.0, vyaw=0.0, duration_s=t)
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
        if self.cli and _env_flag("GO2_USE_SDK_SPECIAL_MOTIONS", default=True):
            self.cli.Dance1()
            self._log("Dance 1 motion")
            return

        self._prepare_locomotion()
        self._timed_move(vx=0.35, vy=0.0, vyaw=0.45, duration_s=1.0)
        self._timed_move(vx=-0.20, vy=0.0, vyaw=-0.45, duration_s=1.0)
        self._timed_move(vx=0.30, vy=0.0, vyaw=-0.45, duration_s=0.9)
        self._timed_move(vx=-0.20, vy=0.0, vyaw=0.45, duration_s=0.8)
        self._log("Dance 1 motion via timed locomotion")

    def dance1(self):
        self.dance()

    def dance2(self):
        if self.cli and _env_flag("GO2_USE_SDK_SPECIAL_MOTIONS", default=True):
            self.cli.Dance2()
            self._log("Dance 2 motion")
            return

        self._prepare_locomotion()
        self._timed_move(vx=0.0, vy=0.0, vyaw=0.6, duration_s=1.0)
        self._timed_move(vx=0.0, vy=0.0, vyaw=-0.6, duration_s=1.0)
        self._timed_move(vx=0.35, vy=0.0, vyaw=0.0, duration_s=0.9)
        self._timed_move(vx=-0.25, vy=0.0, vyaw=0.0, duration_s=0.9)
        self._log("Dance 2 motion via timed locomotion")

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
