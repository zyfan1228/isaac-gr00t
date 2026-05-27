'''
    本地测试使用的客户端
'''

import os
import time
import base64

import cv2
import numpy as np
from loguru import logger
from hong_atom_bridge.communication import rpc_pipe
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def init_rpc_pipe(service_id):
    rpc_client_pipe = rpc_pipe.RPCPipe(service_id, "client")
    return rpc_client_pipe


if __name__ == "__main__":
    os.makedirs("./debug_frames", exist_ok=True)
    client_pipe = init_rpc_pipe("vla_model_infer_service")
    logger.info("Client 启动")

    dataset_path = "/data/fanzhuoyao/data_library/grasp_bottle_0320_20260320_192604_merge"
    dataset = LeRobotDataset(repo_id="lerobot/g1_sft", root=dataset_path)

    hz = 100
    sleep_time = 1.0 / hz

    total_time = 0.0
    i = 0

    for step in range(100):
        frame_data = dataset[step]
        joints_state = frame_data['observation.state'][..., :26].cpu().numpy().tolist()

        # 获取 obs 并使用 base64 序列化
        obs_img_tensor = frame_data['observation.images.cam_left_high']
        obs_img_np = (obs_img_tensor.cpu().numpy() * 255).astype(np.uint8)
        if obs_img_np.shape[0] == 3:
            obs_img_np = np.transpose(obs_img_np, (1, 2, 0))
            obs_img_np = cv2.cvtColor(obs_img_np, cv2.COLOR_RGB2BGR)
        success, encoded_obs_img = cv2.imencode('.png', obs_img_np)

        # for test
        # save_path = f"./debug_frames/frame_step_client.jpg"
        # cv2.imwrite(save_path, obs_img_np)

        if success:
            obs_img_base64 = base64.b64encode(encoded_obs_img.tobytes()).decode('utf-8')
        else:
            pass
        
        # server 端做 tensor 化以及维度对齐
        client_request = {
            "joints_state": joints_state, 
            "obs_img_b64": obs_img_base64, 
            "task": 'grasps the bottle of black cap on the box', # 先 fix
        }
        # breakpoint()
        start_time = time.time()
        response = client_pipe.invoke("infer", client_request)
        latency = (time.time() - start_time) * 1000
        total_time += latency
        i += 1

        action = response.get("action")
        logger.info(f"接收成功 | 延迟{latency:.3f}ms | Chunk 大小: {len(action)}")

        time.sleep(sleep_time)
    
    logger.info(f"平均延迟: {total_time / i}ms")
