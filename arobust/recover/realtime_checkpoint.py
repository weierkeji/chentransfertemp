import getpass
import os
import pickle
import queue
import shutil
import threading
import time
from typing import Any, Dict, Optional

from areal.api.cli_args import RecoverConfig
from realhf.base import logging

logger = logging.getLogger("realtime_checkpoint")


class RealtimeCheckpoint:
    """
    实时异步持久化组件，用于在 rollout 过程中定期保存 buffer 状态。
    不阻塞主流程，使用独立线程执行保存操作。
    """

    def __init__(self, config: RecoverConfig):
        self.config = config
        self.save_queue = queue.Queue(maxsize=10)
        self.saver_thread = None
        self.stop_event = threading.Event()
        self.rollout_buffer = None
        self.dataloader = None
        self.save_interval = 30  # 默认 30 秒

    def get_save_buffer_state_path(self, name: str):
        """获取 buffer 状态保存路径"""
        path = os.path.join(
            f"{self.config.fileroot}/recover/{getpass.getuser()}/{self.config.experiment_name}/{self.config.trial_name}/buffer_states",
            name,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def start_async_saver(
        self,
        rollout_buffer,
        dataloader,
        interval: float = 30,
    ):
        """
        启动后台定时保存线程。
        
        :param rollout_buffer: RolloutBuffer 实例
        :param dataloader: StatefulDataLoader 实例
        :param interval: 保存间隔（秒）
        """
        self.rollout_buffer = rollout_buffer
        self.dataloader = dataloader
        self.save_interval = interval
        self.stop_event.clear()

        self.saver_thread = threading.Thread(
            target=self._async_saver_loop,
            name="RealtimeCheckpointSaver",
            daemon=True,
        )
        self.saver_thread.start()
        logger.info(
            f"Started realtime checkpoint saver with interval={interval}s"
        )

    def stop_async_saver(self):
        """停止后台保存线程"""
        if self.saver_thread and self.saver_thread.is_alive():
            self.stop_event.set()
            self.saver_thread.join(timeout=10)
            logger.info("Stopped realtime checkpoint saver")

    def _async_saver_loop(self):
        """后台保存线程的主循环"""
        last_save_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                if current_time - last_save_time >= self.save_interval:
                    self._save_buffer_state_async()
                    last_save_time = current_time
                
                # 短暂休眠，避免忙等待
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in async saver loop: {e}", exc_info=True)

    def _save_buffer_state_async(self):
        """异步保存 buffer 状态（内部方法）"""
        try:
            if self.rollout_buffer is None or self.dataloader is None:
                return

            buffer_state = self.rollout_buffer.state_dict()
            dataloader_state = self.dataloader.state_dict()
            
            # 使用时间戳作为文件名，确保唯一性
            timestamp = int(time.time())
            buffer_state_name = f"buffer_state_{timestamp}"
            
            # 保存到临时文件
            temp_path = self.get_save_buffer_state_path(f"{buffer_state_name}.tmp")
            final_path = self.get_save_buffer_state_path(buffer_state_name)
            
            state = {
                "buffer_state": buffer_state,
                "dataloader_state": dataloader_state,
                "timestamp": timestamp,
            }
            
            # 写入临时文件
            with open(os.path.join(temp_path, "state.pkl"), "wb") as f:
                pickle.dump(state, f)
            
            # 原子性重命名
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            os.rename(temp_path, final_path)
            
            # 创建或更新 latest 符号链接
            latest_link = self.get_save_buffer_state_path("latest")
            if os.path.exists(latest_link):
                if os.path.islink(latest_link):
                    os.remove(latest_link)
                else:
                    shutil.rmtree(latest_link)
            os.symlink(final_path, latest_link)
            
            logger.info(
                f"Saved buffer state: {buffer_state['current_size']} samples, "
                f"{buffer_state['ready_to_train_sample_num']} ready"
            )
            
            # 清理旧的状态文件（保留最近 3 个）
            self._cleanup_old_states()
            
        except Exception as e:
            logger.error(f"Failed to save buffer state: {e}", exc_info=True)

    def _cleanup_old_states(self):
        """清理旧的状态文件，只保留最近的几个"""
        try:
            base_path = os.path.join(
                f"{self.config.fileroot}/recover/{getpass.getuser()}/{self.config.experiment_name}/{self.config.trial_name}/buffer_states"
            )
            
            if not os.path.exists(base_path):
                return
            
            # 获取所有 buffer_state_* 目录
            state_dirs = [
                d for d in os.listdir(base_path)
                if d.startswith("buffer_state_") and d != "buffer_state_latest"
                and os.path.isdir(os.path.join(base_path, d))
            ]
            
            # 按时间戳排序
            state_dirs.sort(reverse=True)
            
            # 保留最近 3 个，删除其他
            for old_dir in state_dirs[3:]:
                old_path = os.path.join(base_path, old_dir)
                shutil.rmtree(old_path)
                logger.debug(f"Cleaned up old buffer state: {old_dir}")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup old states: {e}")

    def save_final(
        self,
        epoch: int,
        step: int,
        global_step: int,
    ):
        """
        最终保存，在训练结束或异常退出时调用。
        这是一个同步操作。
        """
        try:
            if self.rollout_buffer is None or self.dataloader is None:
                logger.warning("No rollout_buffer or dataloader to save")
                return

            buffer_state = self.rollout_buffer.state_dict()
            dataloader_state = self.dataloader.state_dict()
            
            final_state_name = "buffer_state_final"
            final_path = self.get_save_buffer_state_path(final_state_name)
            
            # 清空目标目录
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            os.makedirs(final_path, exist_ok=True)
            
            state = {
                "buffer_state": buffer_state,
                "dataloader_state": dataloader_state,
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "timestamp": int(time.time()),
            }
            
            with open(os.path.join(final_path, "state.pkl"), "wb") as f:
                pickle.dump(state, f)
            
            logger.info(
                f"Saved final buffer state at epoch={epoch}, step={step}, "
                f"global_step={global_step}, samples={buffer_state['current_size']}"
            )
            
        except Exception as e:
            logger.error(f"Failed to save final buffer state: {e}", exc_info=True)

    @staticmethod
    def load_latest_buffer_state(config: RecoverConfig) -> Optional[Dict[str, Any]]:
        """
        加载最新的 buffer 状态。
        
        :param config: RecoverConfig 实例
        :return: 包含 buffer_state 和 dataloader_state 的字典，如果不存在则返回 None
        """
        try:
            latest_path = os.path.join(
                f"{config.fileroot}/recover/{getpass.getuser()}/{config.experiment_name}/{config.trial_name}/buffer_states",
                "latest",
                "state.pkl"
            )
            
            if not os.path.exists(latest_path):
                logger.info("No latest buffer state found")
                return None
            
            with open(latest_path, "rb") as f:
                state = pickle.load(f)
            
            logger.info(
                f"Loaded latest buffer state: {state['buffer_state']['current_size']} samples"
            )
            return state
            
        except Exception as e:
            logger.warning(f"Failed to load latest buffer state: {e}")
            return None

