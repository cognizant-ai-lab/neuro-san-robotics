
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

        if action == "stand_up":
            go2.stand_up()
            logging.info("GO2 standing up...")

        else:
            return "cannot perform action"

