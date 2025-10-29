
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
import logging
from neuro_san.interfaces.coded_tool import CodedTool
from coded_tools.unigo2.go2_macros import Go2Macros


class RobotMacros(CodedTool):
    """
    CodedTool implementation of robot macros.
    """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:

        action: str = args.get("action")
        if action is None or not isinstance(action, str):
            return "Don't understand non-string actions"

        go2 = Go2Macros()

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
        elif action == "rise_sit":
            go2.rise_sit()
            logging.info("===== GO2 rise sit...")

        # Orientation / Look direction
        elif action == "euler":
            roll = args.get("roll", 0.0)
            pitch = args.get("pitch", 0.0)
            yaw = args.get("yaw", 0.0)
            go2.euler(roll=roll, pitch=pitch, yaw=yaw)
            logging.info(f"===== GO2 euler (roll={roll}, pitch={pitch}, yaw={yaw})...")
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
            logging.info(f"===== GO2 move (vx={vx}, vy={vy}, vyaw={vyaw})...")
        elif action == "step_forward":
            vx = args.get("vx", 0.1)
            t = args.get("t", 1.0)
            go2.step_forward(vx=vx, t=t)
            logging.info(f"===== GO2 stepping forward (vx={vx}, t={t})...")
        elif action == "step_backward":
            vx = args.get("vx", -0.1)
            t = args.get("t", 1.0)
            go2.step_backward(vx=vx, t=t)
            logging.info(f"===== GO2 stepping backward (vx={vx}, t={t})...")
        elif action == "speed_level":
            level = args.get("level", 0)
            go2.speed_level(level)
            logging.info(f"===== GO2 speed level set to {level}...")

        # Special motions / Expressions
        elif action == "shake" or action == "hello":
            go2.shake()
            logging.info("===== GO2 shaking/hello...")
        elif action == "stretch":
            go2.stretch()
            logging.info("===== GO2 stretching...")
        elif action == "content":
            go2.content()
            logging.info("===== GO2 content...")
        elif action == "dance" or action == "dance1":
            go2.dance1()
            logging.info("===== GO2 dance 1...")
        elif action == "dance2":
            go2.dance2()
            logging.info("===== GO2 dance 2...")
        elif action == "pose":
            flag = args.get("flag", True)
            go2.pose(flag)
            logging.info(f"===== GO2 pose (flag={flag})...")
        elif action == "scrape":
            go2.scrape()
            logging.info("===== GO2 scrape...")
        elif action == "heart_pose" or action == "heart":
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
        elif action == "backflip" or action == "back_flip":
            go2.backflip()
            logging.info("===== GO2 backflip...")
        elif action == "hand_stand":
            flag = args.get("flag", True)
            go2.hand_stand(flag)
            logging.info(f"===== GO2 hand stand (flag={flag})...")

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
            logging.info(f"===== GO2 free bound (flag={flag})...")
        elif action == "free_jump":
            flag = args.get("flag", True)
            go2.free_jump(flag)
            logging.info(f"===== GO2 free jump (flag={flag})...")
        elif action == "free_avoid":
            flag = args.get("flag", True)
            go2.free_avoid(flag)
            logging.info(f"===== GO2 free avoid (flag={flag})...")
        elif action == "classic_walk":
            flag = args.get("flag", True)
            go2.classic_walk(flag)
            logging.info(f"===== GO2 classic walk (flag={flag})...")
        elif action == "walk_upright":
            flag = args.get("flag", True)
            go2.walk_upright(flag)
            logging.info(f"===== GO2 walk upright (flag={flag})...")
        elif action == "cross_step":
            flag = args.get("flag", True)
            go2.cross_step(flag)
            logging.info(f"===== GO2 cross step (flag={flag})...")

        # Configuration and settings
        elif action == "switch_joystick":
            on = args.get("on", True)
            go2.switch_joystick(on)
            logging.info(f"===== GO2 switch joystick (on={on})...")
        elif action == "auto_recovery_set":
            enabled = args.get("enabled", True)
            go2.auto_recovery_set(enabled)
            logging.info(f"===== GO2 auto recovery set (enabled={enabled})...")
        elif action == "auto_recovery_get":
            code, data = go2.auto_recovery_get()
            logging.info(f"===== GO2 auto recovery get: {code}, {data}...")
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
