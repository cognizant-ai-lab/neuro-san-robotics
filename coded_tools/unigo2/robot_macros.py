
# Copyright (C) 2023-2025 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# neuro-san SDK Software in commercial settings.
#
# END COPYRIGHT

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
import logging
import threading
from neuro_san.interfaces.coded_tool import CodedTool
from coded_tools.unigo2.go2_macros import Go2Macros


# Global deferred action queue - actions are queued here and executed later
# This allows speech to complete before robot performs physical actions
_deferred_actions: List[Tuple[str, Dict[str, Any]]] = []
_deferred_actions_lock = threading.Lock()


def queue_deferred_action(action: str, args: Dict[str, Any]) -> None:
    """Queue an action to be executed later."""
    with _deferred_actions_lock:
        _deferred_actions.append((action, args))
        logging.info("Queued deferred action: %s", action)


def execute_deferred_actions() -> List[str]:
    """
    Execute all queued deferred actions and clear the queue.

    Returns a list of results from each action execution.
    This should be called after speech completes to ensure proper ordering.
    """
    with _deferred_actions_lock:
        actions_to_execute = list(_deferred_actions)
        _deferred_actions.clear()

    if not actions_to_execute:
        return []

    results = []
    go2 = Go2Macros()

    for action, args in actions_to_execute:
        logging.info("Executing deferred action: %s", action)
        result = _execute_single_action(go2, action, args)
        results.append(result)

    return results


def _execute_single_action(go2: Go2Macros, action: str, args: Dict[str, Any]) -> str:
    """Execute a single robot action. Internal helper function."""
    action = action.lower()

    # Basic motions
    if action == "damp":
        go2.damp()
        logging.info("===== GO2 damp...")
    elif action == "balance_stand":
        go2.balance_stand()
        logging.info("===== GO2 balance stand...")
    elif action == "stop_move":
        go2.stop_move()
        logging.info("===== GO2 stop move...")
    elif action == "stand_up":
        go2.stand_up()
        logging.info("===== GO2 standing up...")
    elif action == "lie_down":
        go2.lie_down()
        logging.info("===== GO2 lying down...")
    elif action == "recovery_stand":
        go2.recovery_stand()
        logging.info("===== GO2 recovery stand...")
    elif action == "sit":
        go2.sit()
        logging.info("===== GO2 sitting...")
    elif action == "sit_rise":
        go2.sit()
        logging.info("===== GO2 sitting...")
    elif action == "rise_sit":
        go2.rise_sit()
        logging.info("===== GO2 rise sit...")

    # Orientation / Look direction
    elif action == "euler":
        roll = args.get("roll", 0.0)
        pitch = args.get("pitch", 0.0)
        yaw = args.get("yaw", 0.0)
        go2.euler(roll=roll, pitch=pitch, yaw=yaw)
        logging.info("===== GO2 euler (roll=%s, pitch=%s, yaw=%s)...", roll, pitch, yaw)
    elif action == "look_left":
        go2.look_left()
        logging.info("===== GO2 looking left...")
    elif action == "look_right":
        go2.look_right()
        logging.info("===== GO2 looking right...")

    # Locomotion
    elif action == "move":
        vx = args.get("vx", 0.0)
        vy = args.get("vy", 0.0)
        vyaw = args.get("vyaw", 0.0)
        go2.move(vx=vx, vy=vy, vyaw=vyaw)
        logging.info("===== GO2 move (vx=%s, vy=%s, vyaw=%s)...", vx, vy, vyaw)
    elif action == "step_forward":
        vx = args.get("vx", 0.3)
        t = args.get("t", 3.0)
        go2.step_forward(vx=vx, t=t)
        logging.info("===== GO2 stepping forward (vx=%s, t=%s)...", vx, t)
    elif action == "step_backward":
        vx = args.get("vx", -0.3)
        t = args.get("t", 3.0)
        go2.step_backward(vx=vx, t=t)
        logging.info("===== GO2 stepping backward (vx=%s, t=%s)...", vx, t)
    elif action == "speed_level":
        level = args.get("level", 0)
        go2.speed_level(level)
        logging.info("===== GO2 speed level set to %s...", level)

    # Special motions / Expressions
    elif action in ("shake", "hello"):
        go2.shake()
        logging.info("===== GO2 shaking/hello...")
    elif action == "stretch":
        go2.stretch()
        logging.info("===== GO2 stretching...")
    elif action == "content":
        go2.content()
        logging.info("===== GO2 content...")
    elif action in ("dance", "dance1"):
        go2.dance1()
        logging.info("===== GO2 dance 1...")
    elif action == "dance2":
        go2.dance2()
        logging.info("===== GO2 dance 2...")
    elif action == "pose":
        flag = args.get("flag", True)
        go2.pose(flag)
        logging.info("===== GO2 pose (flag=%s)...", flag)
    elif action == "scrape":
        go2.scrape()
        logging.info("===== GO2 scrape...")
    elif action in ("heart_pose", "heart"):
        go2.heart_pose()
        logging.info("===== GO2 heart pose...")

    # Flips and acrobatics
    elif action == "front_flip":
        go2.front_flip()
        logging.info("===== GO2 front flip...")
    elif action == "front_jump":
        go2.front_jump()
        logging.info("===== GO2 front jump...")
    elif action == "front_pounce":
        go2.front_pounce()
        logging.info("===== GO2 front pounce...")
    elif action == "left_flip":
        go2.left_flip()
        logging.info("===== GO2 left flip...")
    elif action in ("backflip", "back_flip"):
        go2.backflip()
        logging.info("===== GO2 backflip...")
    elif action == "hand_stand":
        flag = args.get("flag", True)
        go2.hand_stand(flag)
        logging.info("===== GO2 hand stand (flag=%s)...", flag)

    # Gait and walking modes
    elif action == "static_walk":
        go2.static_walk()
        logging.info("===== GO2 static walk...")
    elif action == "trot_run":
        go2.trot_run()
        logging.info("===== GO2 trot run...")
    elif action == "free_walk":
        go2.free_walk()
        logging.info("===== GO2 free walk...")
    elif action == "free_bound":
        flag = args.get("flag", True)
        go2.free_bound(flag)
        logging.info("===== GO2 free bound (flag=%s)...", flag)
    elif action == "free_jump":
        flag = args.get("flag", True)
        go2.free_jump(flag)
        logging.info("===== GO2 free jump (flag=%s)...", flag)
    elif action == "free_avoid":
        flag = args.get("flag", True)
        go2.free_avoid(flag)
        logging.info("===== GO2 free avoid (flag=%s)...", flag)
    elif action == "classic_walk":
        flag = args.get("flag", True)
        go2.classic_walk(flag)
        logging.info("===== GO2 classic walk (flag=%s)...", flag)
    elif action == "walk_upright":
        flag = args.get("flag", True)
        go2.walk_upright(flag)
        logging.info("===== GO2 walk upright (flag=%s)...", flag)
    elif action == "cross_step":
        flag = args.get("flag", True)
        go2.cross_step(flag)
        logging.info("===== GO2 cross step (flag=%s)...", flag)

    # Configuration and settings
    elif action == "switch_joystick":
        on = args.get("on", True)
        go2.switch_joystick(on)
        logging.info("===== GO2 switch joystick (on=%s)...", on)
    elif action == "auto_recovery_set":
        enabled = args.get("enabled", True)
        go2.auto_recovery_set(enabled)
        logging.info("===== GO2 auto recovery set (enabled=%s)...", enabled)
    elif action == "auto_recovery_get":
        code, data = go2.auto_recovery_get()
        logging.info("===== GO2 auto recovery get: %s, %s...", code, data)
        return f"Auto recovery: code={code}, data={data}"
    elif action == "switch_avoid_mode":
        go2.switch_avoid_mode()
        logging.info("===== GO2 switch avoid mode...")

    # Shutdown
    elif action == "shutdown":
        go2.shutdown()
        logging.info("===== GO2 shutting down...")

    else:
        return f"Unknown action: {action}"

    return f"Action '{action}' completed successfully"


class RobotMacros(CodedTool):
    """
    CodedTool implementation of robot macros.
    
    Actions are queued for deferred execution to ensure speech completes
    before physical robot actions are performed. Call execute_deferred_actions()
    after speech to execute the queued actions.
    """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:

        action: str = args.get("action")
        if action is None or not isinstance(action, str):
            return "Don't understand non-string actions"

        action_lower = action.lower()

        # Validate the action is known before queueing
        known_actions = {
            "damp", "balance_stand", "stop_move", "stand_up", "lie_down",
            "recovery_stand", "sit", "rise_sit", "euler", "look_left",
            "look_right", "move", "step_forward", "step_backward", "speed_level",
            "shake", "hello", "stretch", "content", "dance", "dance1", "dance2",
            "pose", "scrape", "heart_pose", "heart", "front_flip", "front_jump",
            "front_pounce", "left_flip", "backflip", "back_flip", "hand_stand",
            "static_walk", "trot_run", "free_walk", "free_bound", "free_jump",
            "free_avoid", "classic_walk", "walk_upright", "cross_step",
            "switch_joystick", "auto_recovery_set", "auto_recovery_get",
            "switch_avoid_mode", "shutdown"
        }

        if action_lower not in known_actions:
            return f"Unknown action: {action}"

        # Queue the action for deferred execution (after speech completes)
        queue_deferred_action(action_lower, dict(args))

        return f"Action '{action}' queued for execution after speech"
