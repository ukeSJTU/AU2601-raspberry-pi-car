# config/config_manager.py

import yaml
import os
import logging
from typing import Dict, Any, Optional


class ConfigManager:
    """系统配置管理器"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件未找到: {path}")

        with open(path, "r", encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
                self._validate_config(config)
                return config
            except yaml.YAMLError as e:
                raise ValueError(f"配置文件格式错误: {e}")

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """验证配置项"""
        required_sections = ["perception", "planning", "control", "vehicle", "system"]

        for section in required_sections:
            if section not in config:
                raise ValueError(f"缺少必要配置段: {section}")

    def _setup_logging(self) -> None:
        """配置日志系统"""
        log_config = self.config.get("system", {}).get("logging", {})

        # 创建日志目录
        log_dir = log_config.get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)

        # 配置日志格式
        log_format = "%(asctime)s [%(levelname)s] %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

        # 设置日志级别
        log_level = getattr(logging, log_config.get("level", "INFO").upper())

        # 配置文件日志
        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(
                    os.path.join(log_dir, "system.log"), encoding="utf-8"
                ),
                logging.StreamHandler(),  # 同时输出到控制台
            ],
        )

    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """获取指定模块的配置"""
        return self.config.get(module_name, {})

    def update_config(self, updates: Dict[str, Any]) -> None:
        """更新配置（仅运行时有效）"""
        for key, value in updates.items():
            if key in self.config:
                self.config[key].update(value)


# debug/data_logger.py

import numpy as np
import cv2
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


class DataLogger:
    """数据记录器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_dir = config.get("log_dir", "logs/data")
        self.record_video = config.get("record_video", False)
        self.record_data = config.get("record_data", True)

        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化视频写入器
        self.video_writer = None
        if self.record_video:
            self._init_video_writer()

        # 初始化数据日志文件
        self.data_file = None
        if self.record_data:
            self._init_data_log()

    def _init_video_writer(self) -> None:
        """初始化视频记录"""
        video_path = os.path.join(
            self.log_dir, f'drive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            self.config.get("video_fps", 30),
            (self.config.get("frame_width", 640), self.config.get("frame_height", 480)),
        )

    def _init_data_log(self) -> None:
        """初始化数据日志"""
        data_path = os.path.join(
            self.log_dir, f'data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
        )
        self.data_file = open(data_path, "w", encoding="utf-8")

    def log_data(self, data: Dict[str, Any]) -> None:
        """记录系统数据"""
        if not self.record_data:
            return

        # 添加时间戳
        data["timestamp"] = datetime.now().isoformat()

        # 转换numpy数组为列表
        processed_data = self._process_data_for_json(data)

        # 写入数据
        self.data_file.write(json.dumps(processed_data, ensure_ascii=False) + "\n")
        self.data_file.flush()

        # 记录关键信息到日志
        if "scene" in data:
            logging.info(
                f'场景: {data["scene"]}, '
                f'行为: {data.get("behavior", {}).get("action", "unknown")}'
            )

    def _process_data_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据使其可JSON序列化"""
        processed = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                processed[key] = value.tolist()
            elif isinstance(value, (int, float, str, bool, list, dict)):
                processed[key] = value
            else:
                processed[key] = str(value)
        return processed

    def save_video(self, frame: np.ndarray, features: Dict[str, Any]) -> None:
        """保存带标注的视频帧"""
        if not self.record_video:
            return

        # 添加特征可视化
        annotated_frame = self._annotate_frame(frame.copy(), features)

        # 记录视频帧
        self.video_writer.write(annotated_frame)

    def _annotate_frame(
        self, frame: np.ndarray, features: Dict[str, Any]
    ) -> np.ndarray:
        """在视频帧上添加特征标注"""
        # 添加场景信息
        if "scene" in features:
            cv2.putText(
                frame,
                f'场景: {features["scene"]}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # 添加行为信息
        if "behavior" in features:
            cv2.putText(
                frame,
                f'行为: {features["behavior"].get("action", "unknown")}',
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        return frame

    def close(self) -> None:
        """关闭日志记录器"""
        if self.video_writer:
            self.video_writer.release()

        if self.data_file:
            self.data_file.close()
