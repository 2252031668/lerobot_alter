from dataclasses import dataclass
from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_openarm_vr_leader")
@dataclass
class BiOpenarmVRLeaderConfig(TeleoperatorConfig):

    ROOT_PATH: str = "/home/wxx/PycharmProjects/lerobot4_alter/src/lerobot/teleoperators/bi_openarm_vr_leader"
    https_port: int =8889
    websocket_port: int = 8890    # 如果修改需要同时修改vr_app.js
    host_ip: str = "0.0.0.0"

